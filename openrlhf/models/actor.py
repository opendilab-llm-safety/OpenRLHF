from typing import Optional, Tuple, Union, List

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
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
        super().__init__()
        self.model_type = model_type
        
        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

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

            if model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrain_or_model,
                    torch_dtype=torch.bfloat16 if bf16 else "auto",
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                    quantization_config=nf4_config,
                    device_map=device_map,
                )
            elif model_type == "qwen2_vl":
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    pretrain_or_model,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16 if bf16 else "auto",
                    device_map=device_map,
                )
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
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        if self.model_type == "qwen2_vl" and 'images' in kwargs and kwargs['images'] is not None:
            # Prepare batch data for vision-language model
            batch_size = input_ids.size(0)
            print(f"\n=== Qwen2 VL Generate ===")
            print(f"Batch size: {batch_size}")
            print(f"Images available: {len(kwargs['images'])}")
            
            assert len(kwargs['images']) == batch_size, \
                f"Images count ({len(kwargs['images'])}) must match batch size ({batch_size})"
            
            # Prepare batch messages
            batch_messages = []
            for i in range(batch_size):
                # Create message with system role
                message = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": []}
                ]
                
                # Add images if available
                if kwargs['images'] is not None:
                    img_paths = kwargs['images'][i] if isinstance(kwargs['images'][i], list) else [kwargs['images'][i]]
                    for img_path in img_paths:
                        message[1]["content"].append({
                            "type": "image",
                            "image": img_path
                        })
                
                # Add text
                text = self.processor.decode(input_ids[i])
                message[1]["content"].append({
                    "type": "text",
                    "text": text
                })
                batch_messages.append(message)

            # Process batch inputs
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages
            ]
            image_inputs, video_inputs = process_vision_info([msg[1] for msg in batch_messages])
            model_inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(input_ids.device)

            # Configure generation arguments
            gen_kwargs = {
                "do_sample": True,
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 50),
                "min_new_tokens": max(kwargs.get("min_new_tokens", 1), 1),
                "max_new_tokens": min(kwargs.get("max_new_tokens", 1024), 1024),
                "eos_token_id": kwargs.get("eos_token_id"),
                "pad_token_id": kwargs.get("pad_token_id"),
                "use_cache": True,
            }

            # Handle beam search settings
            num_beams = kwargs.get("num_beams", 1)
            if num_beams > 1:
                gen_kwargs.update({
                    "num_beams": num_beams,
                    "early_stopping": True,
                    "length_penalty": 1.0,
                    "no_repeat_ngram_size": 3,
                    "temperature": min(gen_kwargs["temperature"], 0.7),
                })
            else:
                gen_kwargs.update({
                    "num_beams": 1,
                    "max_time": 30.0,
                })

            # Update model inputs with generation settings
            model_inputs.update(gen_kwargs)
            
            # Generate sequences
            sequences = self.model.generate(**model_inputs)

            # Process sequences
            input_length = model_inputs.input_ids.size(1)
            generated_length = sequences.size(1)
            
            if generated_length <= input_length:
                raise ValueError(
                    f"Generated sequence ({generated_length}) must be longer than "
                    f"input ({input_length}). Check generation parameters."
                )
        else:
            # Regular language model generation
            generate_args = {
                "input_ids": input_ids,
                "top_k": kwargs.get("top_k", None),
                "top_p": kwargs.get("top_p", None),
                "do_sample": kwargs.get("do_sample", True),
                "temperature": kwargs.get("temperature", 1),
                "use_cache": True,
                "attention_mask": kwargs.get("attention_mask"),
                "eos_token_id": kwargs.get("eos_token_id"),
                "pad_token_id": kwargs.get("pad_token_id"),
                "min_new_tokens": kwargs.get("min_new_tokens", 1),
            }
            
            if kwargs.get("max_new_tokens", None):
                generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
            if kwargs.get("max_length", None):
                generate_args["max_length"] = kwargs.get("max_length")

            sequences = self.model.generate(**generate_args)

        # Process sequences
        eos_token_id = kwargs.get("eos_token_id")
        pad_token_id = kwargs.get("pad_token_id") 
        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        """Process generated sequences to create attention and action masks."""
        print(f"\n=== Processing Sequences ===")
        print(f"Sequences shape: {sequences.shape}")
        print(f"Input length: {input_len}")
        
        # Validate sequence length
        if sequences.size(1) <= input_len:
            raise ValueError(f"Generated sequence length ({sequences.size(1)}) must be greater than input length ({input_len})")

        # Create attention mask
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # Find EOS positions
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # Process token positions
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)
        
        # Create action mask for response tokens
        response_length = sequences.size(1) - input_len + 1
        if response_length <= 0:
            raise ValueError(f"No response tokens available: sequence_length={sequences.size(1)}, input_length={input_len}")
            
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        
        # Always set first response token position to 1 if response exists
        if action_mask.size(1) > 0:
            action_mask[:, 0] = 1
        else:
            raise ValueError("Action mask has zero width - no valid response tokens")

        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
        images: Optional[List[str]] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Returns action log probs"""
        if self.model_type == "qwen2_vl" and images is not None:
            # Create batch messages
            batch_messages = []
            for i in range(sequences.size(0)):
                message = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": []}
                ]
                
                # Add images
                if images[i]:
                    img_paths = images[i] if isinstance(images[i], list) else [images[i]]
                    for img_path in img_paths:
                        message[1]["content"].append({
                            "type": "image",
                            "image": img_path
                        })
                
                # Add text
                text = self.processor.decode(sequences[i])
                message[1]["content"].append({
                    "type": "text",
                    "text": text
                })
                batch_messages.append(message)

            # Process inputs
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages
            ]
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info([msg[1] for msg in batch_messages])
            
            # Create model inputs
            model_inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(sequences.device)

            # Add additional Qwen2-VL specific parameters
            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw
            if rope_deltas is not None:
                model_inputs["rope_deltas"] = rope_deltas

            # Forward pass with additional parameters
            output = self.model(
                **model_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )
        else:
            # Non-vision model or no images provided
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

            # Prepare model inputs
            model_inputs = {
                "input_ids": sequences,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "output_attentions": False,
                "output_hidden_states": False,
                "return_dict": True
            }

            # Add optional parameters if provided
            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw
            if rope_deltas is not None:
                model_inputs["rope_deltas"] = rope_deltas

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
