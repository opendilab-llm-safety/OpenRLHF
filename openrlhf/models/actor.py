from typing import Optional, Tuple, Union, List

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .ring_attn_utils import convert_ring_attn_params
from .utils import log_probs_from_logits, reset_position_ids


class Actor(nn.Module):
    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        model_type="causal_lm",
        **kwargs,
    ) -> None:
        """Initialize Actor model.
        
        Args:
            pretrain_or_model: Model name or pretrained model
            use_flash_attention_2: Enable Flash Attention 2.0
            bf16: Use bfloat16 precision
            load_in_4bit: Load model in 4-bit precision
            lora_rank: LoRA adaptation rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Target modules for LoRA
            ds_config: DeepSpeed configuration
            device_map: Device mapping for model
            packing_samples: Enable sample packing
            model_type: Type of model ("causal_lm", "qwen2_vl", or "vision_lm")
        """
        super().__init__()
        self.model_type = model_type
        
        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Initialize configs
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            # Initialize model based on type
            model_kwargs = {
                "trust_remote_code": True,
                "attn_implementation": attn_implementation,
                "quantization_config": nf4_config,
                "torch_dtype": torch.bfloat16 if bf16 else "auto",
                "device_map": device_map,
            }

            if model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrain_or_model,
                    **model_kwargs
                )
            elif model_type in ["qwen2_vl", "vision_lm"]:
                if model_type == "qwen2_vl":
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        pretrain_or_model,
                        **model_kwargs
                    )
                else:
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        pretrain_or_model,
                        **model_kwargs
                    )
                # Initialize processor for vision-language models
                self.processor = AutoProcessor.from_pretrained(pretrain_or_model)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # LoRA and other configurations...
            if lora_rank > 0:
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            self.model.config.use_cache = False
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:
        """Generate sequences with optional vision input.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            pixel_values: Optional image pixel values
            image_grid_thw: Optional image grid dimensions 
            **kwargs: Additional generation parameters
            
        Returns:
            tuple: (sequences, attention_mask[, action_mask])
        """
        print("\n=== Generate Method ===")
        print(f"Input IDs shape: {input_ids.shape}")
        if pixel_values is not None:
            print(f"Pixel values shape: {pixel_values.shape}")
        if image_grid_thw is not None:
            print(f"Image grid shape: {image_grid_thw.shape}")
        
        # Validate inputs for vision models
        if self.model_type in ["qwen2_vl", "vision_lm"]:
            print(f"\nModel type: {self.model_type}")
            if pixel_values is not None:
                # Validate batch sizes match
                batch_size = input_ids.size(0)
                if len(pixel_values) != batch_size:
                    raise ValueError(
                        f"Batch size mismatch: input_ids({batch_size}) vs "
                        f"pixel_values({len(pixel_values)})"
                    )
                if image_grid_thw is not None and len(image_grid_thw) != batch_size:
                    raise ValueError(
                        f"Batch size mismatch: input_ids({batch_size}) vs "
                        f"image_grid_thw({len(image_grid_thw)})"
                    )

        # Get base generation configuration
        gen_config = self._get_generation_config(**kwargs)
        print("\nGeneration config:", gen_config)

        # Prepare model inputs
        if self.model_type in ["qwen2_vl", "vision_lm"] and pixel_values is not None:
            print("\nPreparing vision-language inputs...")
            model_inputs = self._prepare_vl_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                device=input_ids.device
            )
            model_inputs.update(gen_config)
        else:
            print("\nPreparing text-only inputs...")
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": kwargs.get("attention_mask"),
                **gen_config
            }
            
        # Log important model input shapes
        print("\nModel input shapes:")
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"- {k}: {v.shape}")

        # Generate sequences
        sequences = self.model.generate(**model_inputs)
        
        # Validate generation
        if hasattr(model_inputs, "input_ids"):
            input_length = model_inputs["input_ids"].size(1)
        else:
            input_length = input_ids.size(1)
            
        if sequences.size(1) <= input_length:
            raise ValueError(
                f"Generated sequence ({sequences.size(1)}) must be longer than "
                f"input ({input_length}). Check generation parameters."
            )

        # Process and return sequences
        return self.process_sequences(
            sequences=sequences,
            input_len=input_length,
            eos_token_id=kwargs.get("eos_token_id"),
            pad_token_id=kwargs.get("pad_token_id")
        )

    def _get_generation_config(self, **kwargs):
        """Get common generation configuration."""
        config = {
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "min_new_tokens": max(kwargs.get("min_new_tokens", 1), 1),
            "max_new_tokens": min(kwargs.get("max_new_tokens", 1024), 1024),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "use_cache": True,
        }

        # Configure beam search if requested
        num_beams = kwargs.get("num_beams", 1)
        if num_beams > 1:
            config.update({
                "num_beams": num_beams,
                "early_stopping": True,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 3,
                "temperature": min(config["temperature"], 0.7),
            })
        else:
            config.update({
                "num_beams": 1,
                "max_time": 30.0,
            })

        return config

    def _prepare_vl_inputs(self, input_ids, pixel_values, image_grid_thw, device):
        """
        Prepare inputs for vision-language model.
        Handles different vision-language model formats and ensures proper batching.
        """
        print("\n=== Preparing Vision-Language Inputs ===")
        print(f"Input IDs shape: {input_ids.shape}")
        if pixel_values is not None:
            print(f"Pixel values shape: {pixel_values.shape}")
        if image_grid_thw is not None:
            print(f"Image grid shape: {image_grid_thw.shape}")

        batch_size = input_ids.size(0)
        
        # Validate vision inputs
        if pixel_values is not None:
            if len(pixel_values) != batch_size:
                raise ValueError(f"Batch size mismatch: input_ids({batch_size}) vs pixel_values({len(pixel_values)})")
            if image_grid_thw is not None and len(image_grid_thw) != batch_size:
                raise ValueError(f"Batch size mismatch: input_ids({batch_size}) vs image_grid_thw({len(image_grid_thw)})")
        
        # Create batch messages with proper format handling
        batch_messages = []
        for i in range(batch_size):
            # Basic message structure
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": []}
            ]
            
            # Add vision content if available
            if pixel_values is not None:
                vision_content = {
                    "type": "image",
                    "pixel_values": pixel_values[i]
                }
                # Only add image_grid_thw if it exists
                if image_grid_thw is not None:
                    vision_content["image_grid_thw"] = image_grid_thw[i]
                message[1]["content"].append(vision_content)
            
            # Add text content
            try:
                text = self.processor.decode(input_ids[i], skip_special_tokens=False)
                message[1]["content"].append({
                    "type": "text",
                    "text": text
                })
            except Exception as e:
                print(f"Warning: Error decoding text for batch item {i}: {e}")
                print(f"Input tokens: {input_ids[i]}")
                raise
            
            batch_messages.append(message)

        print(f"\nProcessing batch of {len(batch_messages)} messages")
        
        # Process with model-specific template
        try:
            texts = [
                self.processor.apply_chat_template(
                    msg, 
                    tokenize=False,
                    add_generation_prompt=True
                ) for msg in batch_messages
            ]
        except Exception as e:
            print("Error applying chat template:")
            print(f"First message structure: {batch_messages[0]}")
            raise

        # Prepare final inputs
        model_inputs = self.processor(
            text=texts,
            padding=True,
            return_tensors="pt"
        )
        
        # Add vision inputs if available
        if pixel_values is not None:
            model_inputs.update({
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw
            })
        
        print("\nPrepared inputs:")
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"- {k}: shape {v.shape}")
            else:
                print(f"- {k}: type {type(v)}")
                
        return model_inputs.to(device)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        """Process generated sequences to create attention and action masks.
        Handles both text-only and multi-modal sequences."""
        print(f"\n=== Processing Sequences ===")
        print(f"Sequences shape: {sequences.shape}")
        print(f"Input length: {input_len}")
        
        # Validate sequence length
        if sequences.size(1) <= input_len:
            raise ValueError(f"Generated sequence length ({sequences.size(1)}) must be greater than input length ({input_len})")
            
        # Create attention mask based on valid tokens
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)
        
        # Process EOS positions
        eos_positions = (sequences == eos_token_id).nonzero(as_tuple=True)[1]
        if len(eos_positions) > 0:
            # Group EOS positions by batch index
            batch_size = sequences.size(0)
            eos_indices = []
            for i in range(batch_size):
                batch_eos = eos_positions[torch.div(eos_positions, seq_length, rounding_mode='floor') == i]
                if len(batch_eos) > 0:
                    # Take first EOS position for each sequence
                    eos_indices.append(batch_eos[0])
                else:
                    # If no EOS found, use sequence end
                    eos_indices.append(seq_length - 1)
            eos_indices = torch.tensor(eos_indices, device=sequences.device).unsqueeze(1)
        else:
            # If no EOS tokens found, use sequence end positions
            eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)
        
        # Process token positions handling left-padded inputs
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        position_indices = torch.arange(seq_length, device=sequences.device).unsqueeze(0).expand(sequences.size(0), -1)
        attention_mask = (position_indices >= first_token_indices) & (position_indices <= eos_indices)
        attention_mask = attention_mask.to(dtype=torch.long)
        
        # Create action mask for response tokens, handling potential image tokens
        response_start = max(0, input_len - 1)  # Handle cases with image tokens
        state_seq = sequences[:, response_start:-1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        
        # Ensure first response token is always included
        if action_mask.size(1) > 0:
            action_mask[:, 0] = 1
            
            # Handle potential image token positions in action mask
            if self.model_type in ["qwen2_vl", "vision_lm"]:
                # Exclude image token positions from action mask
                image_token_ids = set(self.processor.image_token_map.values()) if hasattr(self, 'processor') else set()
                for token_id in image_token_ids:
                    action_mask &= (state_seq != token_id)
        else:
            raise ValueError("Action mask has zero width - no valid response tokens")
            
        print("\nMask shapes:")
        print(f"Attention mask: {attention_mask.shape}")
        print(f"Action mask: {action_mask.shape}")

        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        return_output=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Returns action log probs"""
        
        # Handle position IDs for both packed and unpacked cases
        if not self.packing_samples:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            if ring_attn_group is not None:
                sequences, attention_mask, position_ids = convert_ring_attn_params(
                    sequences, attention_mask, packed_seq_lens, ring_attn_group
                )
            else:
                position_ids = reset_position_ids(attention_mask)
            attention_mask = None

        print("\n=== Forward Pass ===")
        print(f"Sequences shape: {sequences.shape}")
        if attention_mask is not None:
            print(f"Attention mask shape: {attention_mask.shape}")
        if pixel_values is not None:
            print(f"Pixel values shape: {pixel_values.shape}")
        if image_grid_thw is not None:
            print(f"Image grid shape: {image_grid_thw.shape}")
            
        # Validate vision inputs if needed
        if self.model_type in ["qwen2_vl", "vision_lm"] and pixel_values is not None:
            batch_size = sequences.size(0)
            if len(pixel_values) != batch_size:
                raise ValueError(f"Batch size mismatch: sequences({batch_size}) vs pixel_values({len(pixel_values)})")
            if image_grid_thw is not None and len(image_grid_thw) != batch_size:
                raise ValueError(f"Batch size mismatch: sequences({batch_size}) vs image_grid_thw({len(image_grid_thw)})")

        # Prepare base model inputs
        model_inputs = {
            "input_ids": sequences,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True
        }

        # Add vision-specific inputs if needed
        if self.model_type in ["qwen2_vl", "vision_lm"] and pixel_values is not None:
            vision_inputs = {
                "pixel_values": pixel_values,
            }
            if image_grid_thw is not None:
                vision_inputs["image_grid_thw"] = image_grid_thw
                
            print("\nAdding vision inputs:")
            for k, v in vision_inputs.items():
                print(f"- {k}: shape {v.shape}")
            model_inputs.update(vision_inputs)
            
        # Add optional parameters
        if rope_deltas is not None:
            model_inputs["rope_deltas"] = rope_deltas

        # Forward pass
        output = self.model(**model_inputs)

        output["logits"] = output["logits"].to(torch.float32)

        if num_actions is None:
            assert return_output
            return output

        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if not self.packing_samples:
            action_log_probs = log_probs[:, -num_actions:]
        else:
            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_log_probs = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_log_probs.append(log_probs[:, start:end])
                offset += seq_len
            action_log_probs = torch.cat(action_log_probs, dim=1)

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
