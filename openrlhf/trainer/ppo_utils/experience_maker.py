import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field, InitVar, make_dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data with sequence length and number of actions.
    Left padding is applied to sequences.
    
    All tensors have shape (batch_size, ...) unless specified otherwise.
    Prefixes indicate expected order of arguments.
    
    Basic fields (required):
    1. sequences: (B, S) - Input tokens
    2. action_log_probs: (B, A) - Action log probabilities
    3. values: (B, A) - Value estimates 
    4. attention_mask: (B, S) - Attention mask
    5. action_mask: (B, A) - Action mask
    6. info: Additional metadata
    7. response_length: (B,) - Response lengths
    8. total_length: (B,) - Total sequence lengths
    
    Optional fields:
    - returns: (B, A) - Returns
    - advantages: (B, A) - Advantages
    - kl: (B, A) - KL divergence
    - pixel_values: (B*h, w) - Image pixels
    - image_grid_thws: (B, 3) - Image grid info
    
    B = batch, S = sequence len, A = action len, h/w = image dims
    """
    # Required fields with explicit field definition
    sequences: torch.Tensor = field(default=None)
    action_log_probs: torch.Tensor = field(default=None)
    values: torch.Tensor = field(default=None)
    attention_mask: torch.LongTensor = field(default=None)
    action_mask: torch.BoolTensor = field(default=None)
    info: dict = field(default_factory=dict)
    response_length: torch.Tensor = field(default=None)
    total_length: torch.Tensor = field(default=None)
    
    # Optional fields 
    returns: Optional[torch.Tensor] = field(default=None, compare=False)
    advantages: Optional[torch.Tensor] = field(default=None, compare=False)
    kl: Optional[torch.Tensor] = field(default=None, compare=False)
    pixel_values: Optional[torch.Tensor] = field(default=None, compare=False)
    image_grid_thws: Optional[torch.Tensor] = field(default=None, compare=False)

    def __post_init__(self):
        """Validate required fields after initialization."""
        required_fields = [
            'sequences', 'action_log_probs', 'values', 
            'attention_mask', 'action_mask', 'info',
            'response_length', 'total_length'
        ]
        for field_name in required_fields:
            if getattr(self, field_name) is None:
                raise ValueError(f"Required field '{field_name}' cannot be None")

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensors to specified device."""
        # Create a dict with updated values since the dataclass is frozen
        updates = {}
        
        # Move required fields
        updates["sequences"] = to(self.sequences, device)
        updates["action_log_probs"] = to(self.action_log_probs, device)
        updates["values"] = to(self.values, device)
        updates["attention_mask"] = to(self.attention_mask, device)
        updates["action_mask"] = to(self.action_mask, device)
        updates["info"] = {key: to(value, device) for key, value in self.info.items()} if self.info else {}
        updates["response_length"] = to(self.response_length, device)
        updates["total_length"] = to(self.total_length, device)
        
        # Move optional fields
        updates["returns"] = to(self.returns, device)
        updates["advantages"] = to(self.advantages, device)
        updates["kl"] = to(self.kl, device)
        
        # Handle vision data if present
        if self.pixel_values is not None:
            updates["pixel_values"] = to(self.pixel_values, device)
            updates["image_grid_thws"] = to(self.image_grid_thws, device)
        
        # Create new instance with updated values
        return Experience(**updates)

    def pin_memory(self):
        """Pin memory for faster data transfer."""
        # Create a dict with updated values since the dataclass is frozen
        updates = {}
        
        # Pin required fields
        updates["sequences"] = pin_memory(self.sequences)
        updates["action_log_probs"] = pin_memory(self.action_log_probs)
        updates["values"] = pin_memory(self.values)
        updates["attention_mask"] = pin_memory(self.attention_mask)
        updates["action_mask"] = pin_memory(self.action_mask)
        updates["info"] = {key: pin_memory(value) for key, value in self.info.items()} if self.info else {}
        updates["response_length"] = pin_memory(self.response_length)
        updates["total_length"] = pin_memory(self.total_length)
        
        # Pin optional fields
        updates["returns"] = pin_memory(self.returns)
        updates["advantages"] = pin_memory(self.advantages)
        updates["kl"] = pin_memory(self.kl)
        
        # Handle vision data if present
        if self.pixel_values is not None:
            updates["pixel_values"] = pin_memory(self.pixel_values)
            updates["image_grid_thws"] = pin_memory(self.image_grid_thws)
        
        # Create new instance with updated values
        return Experience(**updates)


@dataclass(frozen=True, order=False)
class Samples:
    """Samples batch supporting both text-only and multi-modal data.
    
    All tensors have shape (batch_size, ...) unless specified otherwise.
    Two formats supported: batched (with padding) or packed (concatenated).
    
    Required fields:
    1. sequences: (B,S)/(1,L) - Token sequences
    2. num_actions: int/(B,) - Response token counts
    3. response_length: (B,) - Response lengths per sample
    4. total_length: (B,) - Total sequence lengths 
    
    Optional fields (based on format and modality):
    - attention_mask: (B,S)/(1,L) - Attention masks
    - action_mask: (B,A)/None - Response masks
    - packed_seq_lens: None/(B,) - Individual lengths when packed
    - pixel_values: (B*h,w) - Image pixels for vision
    - image_grid_thws: (B,3) - Image grid dimensions
    - references: List[str] - Reference answers
    - images: List[str/Tensor] - Source images
    
    B = batch size, S = sequence len, A = action len
    L = total len, h/w = image dimensions
    """
    # Required fields must be listed first
    sequences: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    
    # Optional fields with defaults after required fields
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    packed_seq_lens: Optional[torch.Tensor] = None
    pixel_values: Optional[torch.Tensor] = None  
    image_grid_thws: Optional[torch.Tensor] = None
    references: Optional[List[str]] = None
    images: Optional[List[Union[str, torch.Tensor]]] = None


class NaiveExperienceMaker(ABC):
    """Naive experience maker."""

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        processor=None,
        prompt_max_len: int,
        kl_controller=None,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        """Process text inputs using tokenizer."""
        if not padding:
            # When padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def processor_fn(self, texts, images=None, max_length=None, padding=True, device=None):
        """Process text and image inputs using processor if available.
        Falls back to tokenizer if no processor exists or no images provided."""
        if not hasattr(self, 'processor') or self.processor is None or images is None:
            return self.tokenize_fn(texts, max_length, padding, device)
            
        if not padding:
            # When padding is False, return processed inputs as list
            return self.processor(
                text=texts,
                images=images,
                max_length=max_length, 
                truncation=True,
                return_tensors=None
            )

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True
        )
        
        # Filter non-tensor values before device placement
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
               for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(
        self, 
        all_prompts: Union[str, List[str], Tuple[List[str], List[str], List[str]]], 
        **generate_kwargs
    ) -> List[Experience]:
        """
        Make experiences from prompts and optionally images.
        
        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        
        print(f"\n=== [make_experience_list] Initial Data ===")
        print(f"all_prompts type: {type(all_prompts)}")
        print(f"all_prompts length: {len(all_prompts)}")
        print(f"n_samples_per_prompt: {args.n_samples_per_prompt}")
        
        # Handle prompts, references and images
        print(f"\n=== [make_experience_list] ===")
        print(f"Input all_prompts format: {type(all_prompts)}")
        print(f"First element type: {type(all_prompts[0]) if all_prompts else None}")
        
        print("\n=== Data Format Check ===")
        # 检查输入格式并规范化
        if isinstance(all_prompts, (tuple, list)) and len(all_prompts) > 0:
            # 处理组合数据
            if isinstance(all_prompts[0], list):
                if len(all_prompts) == 3:
                    # 三元组数据: [prompts_list, refs_list, images_list]
                    prompts_list = all_prompts[0]
                    refs_list = all_prompts[1]
                    imgs_list = all_prompts[2]
                    
                    if not (len(prompts_list) == len(refs_list) == len(imgs_list)):
                        raise ValueError(f"Length mismatch: prompts({len(prompts_list)}), refs({len(refs_list)}), images({len(imgs_list)})")
                    print(f"Triple data format - {len(prompts_list)} samples with images")
                    all_prompts, references, images = prompts_list, refs_list, imgs_list
                    
                elif len(all_prompts) == 2:
                    # 双元组数据: [prompts_list, refs_list]
                    prompts_list = all_prompts[0]
                    refs_list = all_prompts[1]
                    
                    if len(prompts_list) != len(refs_list):
                        raise ValueError(f"Length mismatch: prompts({len(prompts_list)}), refs({len(refs_list)})")
                    print(f"Paired data format - {len(prompts_list)} samples")
                    all_prompts, references, images = prompts_list, refs_list, None
                    
                else:
                    raise ValueError(f"Expected 2 or 3 components, got {len(all_prompts)}")
            else:
                # 单一数据列表
                print(f"Single data format - {len(all_prompts)} samples")
                references, images = None, None
        
            
        # Generate responses with images for multi-modal models
        samples_list = self.generate_samples(all_prompts=all_prompts, references=references, images=images, **generate_kwargs)
        torch.distributed.barrier()

        experiences = []
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples).to_device("cpu"))

        experiences, rewards = self.process_experiences(experiences)

        # Calculate return and advantages
        # index = 0
        for experience, reward in zip(experiences, rewards):
            # index += 1
            # print(f"experience {index}: {experience}")
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unknown advantage_estimator {self.advantage_estimator}")

            # Calculate return info
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # Remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], references: Optional[List[str]] = None, images: Optional[List[str]] = None, **generate_kwargs) -> List[Samples]:
        """Generate samples and return in batches."""
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()

        print("\n=== ExperienceMaker Generate Samples ===")
        print("Initial data:")
        print(f"- Prompts count: {len(all_prompts)}")
        print(f"- References count: {len(references) if references else 0}")
        print(f"- Images count: {len(images) if images else 0}")
        if images:
            print("First few images:")
            for i, img in enumerate(images[:2]):
                print(f"  {i}: {img if isinstance(img, str) else len(img)} images")

        print("\n=== [generate_samples] Initial Data ===")
        print(f"Prompts count: {len(all_prompts)}")
        print(f"References count: {len(references) if references else 0}")
        print(f"Images count: {len(images) if images else 0}")
        print(f"n_samples_per_prompt: {args.n_samples_per_prompt}")

        # 1. 首先验证输入数据的对齐
        if references is not None and len(references) != len(all_prompts):
            raise ValueError(f"References count ({len(references)}) does not match prompts count ({len(all_prompts)})")
        if images is not None and len(images) != len(all_prompts):
            raise ValueError(f"Images count ({len(images)}) does not match prompts count ({len(all_prompts)})")

        # 2. 统一扩展数据
        expanded_data = []
        for idx in range(len(all_prompts)):
            prompt = all_prompts[idx]
            ref = references[idx] if references else None
            img = images[idx] if images else None
            
            for _ in range(args.n_samples_per_prompt):
                expanded_data.append({
                    'prompt': prompt,
                    'reference': ref,
                    'image': img if isinstance(img, str) else (deepcopy(img) if img else None)
                })

        print(f"\n=== After {args.n_samples_per_prompt}x expansion ===")
        print(f"Total samples: {len(expanded_data)}")
                
        # 3. 解包扩展后的数据
        all_prompts = [item['prompt'] for item in expanded_data]
        references = [item['reference'] for item in expanded_data] if references else None
        images = [item['image'] for item in expanded_data] if images else None

        # 验证扩展后的对齐
        sample_counts = {
            'prompts': len(all_prompts),
            'references': len(references) if references else 0,
            'images': len(images) if images else 0
        }
        if len(set(v for v in sample_counts.values() if v > 0)) > 1:
            raise ValueError(f"Data misaligned after expansion: {sample_counts}")
            
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            print(f"\n=== Processing Batch {i//args.micro_rollout_batch_size + 1} ===")
            start_idx = i
            end_idx = i + args.micro_rollout_batch_size
            
            prompts = all_prompts[start_idx:end_idx]
            batch_references = references[start_idx:end_idx] if references else None
            batch_images = images[start_idx:end_idx] if images else None
            
            print(f"\n=== Batch Details ===")
            print(f"- Range: {start_idx} to {end_idx}")
            print(f"- Prompts count: {len(prompts)}")
            print(f"- References count: {len(batch_references) if batch_references else 0}")
            print(f"- Images count: {len(batch_images) if batch_images else 0}")
            
            # 处理图片路径，确保格式统一
            if batch_images:
                batch_images = [[img] if isinstance(img, str) else img for img in batch_images]
                print(f"Sample images:")
                for i, img_list in enumerate(batch_images[:2]):
                    print(f"  Sample {i}: {img_list}")
            
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, images=batch_images, **generate_kwargs)

            samples = Samples(
                # Basic required fields
                sequences=sequences,
                num_actions=action_mask.size(1),
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                
                # Optional fields
                attention_mask=attention_mask,
                action_mask=action_mask,
                packed_seq_lens=None,
                
                # Additional metadata
                references=batch_references,
                images=batch_images
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        if self.strategy.args.perf:
            print("\n=== Make Experience Performance Stats ===")
            print(f"Processing sample with shape: {samples.sequences.shape}")

        # Extract all values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        pixel_values = samples.pixel_values
        image_grid_thws = samples.image_grid_thws

        # Handle vision data type conversion if needed
        vision_inputs = {}
        if pixel_values is not None:
            vision_inputs.update({
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thws
            })
            print("\n=== Vision Inputs ===")
            print(f"pixel_values shape: {pixel_values.shape}")
            print(f"image_grid_thws shape: {image_grid_thws.shape}")
            print(f"num_actions: {num_actions}")

        # Get action log probs
        print("\n=== Actor Forward ===")
        action_log_probs = self.actor(
            sequences=sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            **vision_inputs
        )
        print(f"action_log_probs shape: {action_log_probs.shape}")

        # Get initial model log probs 
        print("\n=== Initial Model Forward ===")
        base_action_log_probs = self.initial_model(
            sequences=sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            **vision_inputs
        )
        print(f"base_action_log_probs shape: {base_action_log_probs.shape}")

        # Get values
        if self.critic is not None:
            print("\n=== Critic Forward ===")
            value = self.critic(
                sequences=sequences,
                num_actions=num_actions,
                attention_mask=attention_mask,
                **vision_inputs
            )
            print(f"value shape: {value.shape if value is not None else None}")
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            # Remote reward computation
            print("\n=== Remote Reward Model Processing ===")
            
            # Prepare sequences
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            print(f"Decoded {len(queries)} sequences")
            
            # Handle references if available
            references = None
            if hasattr(samples, "references") and samples.references is not None:
                references = samples.references
                print(f"Using {len(references)} references for reward computation")
                print("Sample references:")
                for i, ref in enumerate(references[:2]):
                    print(f"  {i}: {ref[:100]}...")
            
            # Prepare vision data if available
            vision_data = None
            if hasattr(samples, 'pixel_values') and samples.pixel_values is not None:
                vision_data = {
                    'pixel_values': samples.pixel_values.cpu(),
                }
                if samples.image_grid_thws is not None:
                    vision_data['image_grid_thw'] = samples.image_grid_thws.cpu()
                print("\nIncluding vision data:")
                print(f"- Pixel values shape: {vision_data['pixel_values'].shape}")
                if 'image_grid_thw' in vision_data:
                    print(f"- Image grid shape: {vision_data['image_grid_thw'].shape}")
            
            # Log request details
            print("\nRequest details:")
            print(f"- Sequence shape: {sequences.shape}")
            print(f"- Queries: {len(queries)}")
            print(f"- References: {len(references) if references else 0}")
            print(f"- Vision inputs: {'Yes' if vision_data else 'No'}")
            print(f"- Remote RM URL: {self.remote_rm_url}")
            
            # Submit reward computation request
            r = remote_rm_fn(
                self.remote_rm_url,
                queries=queries,
                references=references,
                vision_data=vision_data
            ).to(device=action_log_probs.device)
            
            print(f"Reward computation completed, shape: {r.shape}")
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            # Basic required fields (non-default parameters)
            sequences=sequences,
            action_log_probs=action_log_probs,
            values=value,
            attention_mask=attention_mask,
            action_mask=action_mask,
            info=info,
            response_length=samples.response_length,
            total_length=samples.total_length,
            
            # Optional fields (with default values)
            returns=None,
            advantages=None,
            kl=kl,
            pixel_values=pixel_values,
            image_grid_thws=image_grid_thws
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for RLOO
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_prompts, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, **generate_kwargs)

        return self._generate_vllm(all_prompts, **generate_kwargs)

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        
        # Initialize performance tracking if needed
        if self.strategy.args.perf:
            print("\n=== Make Remote Experience Performance Stats ===")
            print(f"Processing sample with shape: {samples.sequences.shape}")

        self.actor.eval()
        device = torch.cuda.current_device()

        # Extract all sample values
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens
        pixel_values = samples.pixel_values
        image_grid_thws = samples.image_grid_thws

        start = time.time()

        # Move tensors to CPU for remote operations
        sequences_cpu = sequences.to("cpu")
        attention_mask_cpu = attention_mask.to("cpu")
        
        # Prepare vision inputs if available
        vision_inputs = {}
        if pixel_values is not None:
            pixel_values_cpu = pixel_values.to("cpu")
            image_grid_thws_cpu = image_grid_thws.to("cpu")
            vision_inputs.update({
                "pixel_values": pixel_values_cpu,
                "image_grid_thw": image_grid_thws_cpu
            })
            print("\n=== Vision Inputs ===")
            print(f"pixel_values shape: {pixel_values.shape}")
            print(f"image_grid_thws shape: {image_grid_thws.shape}")

        # Get action log probs from actor
        print("\n=== Actor Forward ===")
        action_log_probs = self.actor(
            sequences=sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            packed_seq_lens=packed_seq_lens,
            **({k: v.to(device) for k, v in vision_inputs.items()} if vision_inputs else {})
        )
        print(f"action_log_probs shape: {action_log_probs.shape}")

        # Get base action log probs from initial model
        print("\n=== Initial Model Forward ===")
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu,
            num_actions,
            attention_mask_cpu,
            return_output=False,
            ring_attn_group=None,
            packed_seq_lens=packed_seq_lens,
            **vision_inputs
        )
        print(f"Submitted base_action_log_probs request")

        # Get critic value if available
        if self.critic is not None:
            print("\n=== Critic Forward ===")
            value_ref = self.critic.forward.remote(
                sequences_cpu,
                num_actions,
                attention_mask_cpu,
                return_output=False,
                ring_attn_group=None,
                packed_seq_lens=packed_seq_lens,
                **vision_inputs
            )
            print(f"Submitted value request")
            
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # Process rewards
        print("\n=== Processing Rewards ===")
        r_refs = []
        if not self.remote_rm_url:
            # Local reward model
            for rm in self.reward_model:
                # Check if reward model supports vision
                rm_supports_vision = ray.get(rm.has_vision_support.remote()) if hasattr(rm, 'has_vision_support') else False
                print(f"Reward Model vision support: {rm_supports_vision}")
                
                forward_inputs = {
                    "sequences": sequences_cpu,
                    "attention_mask": attention_mask_cpu,
                    "packed_seq_lens": packed_seq_lens
                }
                if rm_supports_vision and vision_inputs:
                    forward_inputs.update(vision_inputs)
                
                r_refs.append(rm.forward.remote(**forward_inputs))
        else:
            # Remote reward model
            if not self.packing_samples:
                queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
            else:
                # Handle packed sequence decoding
                sequences_list = []
                offset = 0
                tokens_list = sequences_cpu.tolist()[0]
                for length in packed_seq_lens:
                    sequences_list.append(tokens_list[offset : offset + length])
                    offset += length
                queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

            # Get references if available
            references = samples.references if hasattr(samples, "references") else None
            if references:
                print(f"Including {len(references)} references for reward computation")

            # Submit remote reward computation requests
            for rm in self.remote_rm_url:
                print(f"Submitting to remote RM: {rm}")
                r = remote_rm_fn_ray.remote(
                    rm, 
                    queries=queries,
                    references=references,
                    vision_data=vision_inputs if vision_inputs else None
                )
                r_refs.append(r)

        actor_value_rm_time = time.time() - start

        # Wait for all remote operations to complete
        print("\n=== Collecting Remote Results ===")
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        # Process results
        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        
        # Move results to correct device
        base_action_log_probs = to(base_action_log_probs, device)
        value = to(value, device) if value is not None else None
        rewards = [to(r, device) for r in rewards]
        
        # Combine rewards if needed
        r = self.reward_fn(rewards) if len(rewards) > 0 and self.reward_fn else rewards[0]

        # Memory management
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])
        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        # Compute KL divergence
        print("\n=== Computing KL Divergence ===")
        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        # Process results based on packing mode
        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)
            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        # Prepare info dictionary
        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        # Record performance stats
        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        # Create and return experience
        print("\n=== Creating Experience ===")
        experience = Experience(
            # Basic required fields (non-default parameters)
            sequences=sequences,
            action_log_probs=action_log_probs,
            values=value,
            attention_mask=attention_mask,
            action_mask=action_mask,
            info=info,
            response_length=samples.response_length,
            total_length=samples.total_length,
            
            # Optional fields (default parameters)
            returns=None,
            advantages=None,
            kl=kl,
            pixel_values=pixel_values,
            image_grid_thws=image_grid_thws
        )

        self.actor.train()  # Reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[str], images: Optional[List[str]] = None, references: Optional[List[str]] = None, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        print("\n=== [_generate_vllm] Initial Data ===")
        print(f"Prompts count: {len(all_prompts)}")
        print(f"Images count: {len(images) if images else 0}")
        print(f"References count: {len(references) if references else 0}")
        if images:
            print(f"First few images: {images[:2]}")

        print("\n=== Initializing VLLM Generation ===")
        # 1. 初始化采样参数
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # 2. Load balancing
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        llms = [self.vllm_engines[rank % len(self.vllm_engines)]] if len(self.vllm_engines) <= world_size else self.vllm_engines[rank::world_size]

        # 3. 格式化和验证数据
        print("\n=== Data Validation ===")
        # 确保图像格式统一(转为列表)
        if images:
            images = [img if isinstance(img, list) else [img] if img else [] for img in images]
            print(f"Images format: {[len(img_list) for img_list in images[:3]]}...")

        # 4. 数据扩展
        expanded_prompts = []
        expanded_images = []
        expanded_refs = []
        
        print("\n=== Expanding Data ===")
        # 确保每个样本生成n_samples_per_prompt个副本,且保持数据对齐
        for idx in range(len(all_prompts)):
            for _ in range(self.strategy.args.n_samples_per_prompt):
                expanded_prompts.append(all_prompts[idx])
                if images:
                    expanded_images.append(deepcopy(images[idx]))
                if references:
                    expanded_refs.append(references[idx])
                    
        print(f"Data sizes after expansion:")
        print(f"- Prompts: {len(expanded_prompts)}")
        print(f"- Images: {len(expanded_images) if expanded_images else 0}")
        print(f"- References: {len(expanded_refs) if expanded_refs else 0}")
        
        # 5. Tokenize prompts
        print("\n=== Tokenizing ===")
        # 5. Tokenize and prepare batches
        print("\n=== Preparing Batches ===")
        all_prompt_token_ids = self.tokenize_fn(expanded_prompts, self.prompt_max_len, padding=False)["input_ids"]
        
        # Calculate batch distribution
        total_samples = len(expanded_prompts)
        samples_per_llm = (total_samples + len(llms) - 1) // len(llms)
        
        print(f"\nBatch Distribution:")
        print(f"Total samples: {total_samples}")
        print(f"LLM engines: {len(llms)}")
        print(f"Samples per LLM: {samples_per_llm}")
        
        # 6. Distribute samples to LLMs
        all_output_refs = []
        for i, llm in enumerate(llms):
            start_idx = i * samples_per_llm
            end_idx = min((i + 1) * samples_per_llm, total_samples)
            
            if start_idx >= end_idx:
                continue
                
            # Prepare batch data
            batch_tokens = all_prompt_token_ids[start_idx:end_idx]
            batch_images = expanded_images[start_idx:end_idx] if expanded_images else None
            
            print(f"\nLLM {i} Batch:")
            print(f"- Range: {start_idx} to {end_idx}")
            print(f"- Sequences: {len(batch_tokens)}")
            print(f"- Images: {len(batch_images) if batch_images else 0}")
            if batch_images:
                print(f"- Images per sample: {[len(img) if img else 0 for img in batch_images[:3]]}...")
            
            # Validate batch alignment
            if batch_images:
                assert len(batch_tokens) == len(batch_images), \
                    f"Batch size mismatch - tokens: {len(batch_tokens)}, images: {len(batch_images)}"
            
            # Submit generation request
            print(f"Submitting to LLM {i}...")
            ref = llm.generate.remote(
                prompt_token_ids=batch_tokens,
                sampling_params=sampling_params,
                images=batch_images,
            )
            all_output_refs.append(ref)

        # 7. Collect results
        print("\n=== Collecting Results ===")
        all_outputs = []
        for i, ref in enumerate(all_output_refs):
            print(f"Getting results from LLM {i}...")
            outputs = ray.get(ref)
            
            # Handle both list and single output cases
            if isinstance(outputs, list):
                all_outputs.extend(outputs)
            else:
                all_outputs.append(outputs)
                
        print(f"Total outputs collected: {len(all_outputs)}")
        print(f"Expected outputs: {total_samples}")
        
        # Validate output count
        assert len(all_outputs) == total_samples, \
            f"Output count mismatch - got: {len(all_outputs)}, expected: {total_samples}"

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                
                # Get current batch images and references
                start_idx = i
                end_idx = i + args.micro_rollout_batch_size
                current_batch_images = all_images[start_idx:end_idx] if all_images else None
                current_batch_references = references[start_idx:end_idx] if references else None

                print(f"\n=== Processing Regular Batch {i//args.micro_rollout_batch_size + 1} ===")
                print(f"Batch size: {len(sequences)}")
                if current_batch_images:
                    print(f"Images per sample: {[len(img_list) for img_list in current_batch_images]}")
                if current_batch_references:
                    print(f"References count: {len(current_batch_references)}")
                
                samples_list.append(
                    Samples(
                        # Basic required fields
                        sequences=sequences,
                        num_actions=action_mask.size(1),
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        
                        # Optional fields
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        packed_seq_lens=None,
                        
                        # Additional metadata
                        references=current_batch_references,
                        images=current_batch_images
                    )
                )
            else:
                # For packing_samples mode
                print(f"\n=== Processing Packed Batch ===")
                batch_start = i
                batch_end = min(i + args.micro_rollout_batch_size, len(all_outputs))
                outputs_batch = all_outputs[batch_start:batch_end]
                print(f"Processing indices {batch_start} to {batch_end} ({len(outputs_batch)} sequences)")
                
                # Get corresponding data for this batch
                current_batch_images = all_images[batch_start:batch_end] if all_images else None
                current_batch_references = references[batch_start:batch_end] if references else None
                
                print("\nProcessing batch data:")
                print(f"Range: {batch_start} to {batch_end}")
                print(f"Output sequences: {len(outputs_batch)}")
                print(f"Images: {len(current_batch_images) if current_batch_images else 0}")
                print(f"References: {len(current_batch_references) if current_batch_references else 0}")
                
                if current_batch_images:
                    print(f"Batch images: {[len(img_list) if img_list else 0 for img_list in current_batch_images]}")
                if current_batch_references:
                    print(f"Batch references: {len(current_batch_references)}")
                
                # Initialize collection lists
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                response_lengths = []
                total_lengths = []
                
                # Process sequences and collect metadata
                for output in outputs_batch:
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    total_len = input_len + output_len
                    
                    # Collect sequence tokens
                    sequence_tokens = output.prompt_token_ids + list(output.outputs[0].token_ids)
                    sequences.extend(sequence_tokens)
                    packed_seq_lens.append(total_len)
                    attention_mask.extend([len(packed_seq_lens)] * total_len)  # Use 1-based index
                    response_lengths.append(output_len)
                    total_lengths.append(total_len)
                
                # Create tensors
                sequences = torch.tensor([sequences], device="cuda")
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                
                print("\nPacked batch stats:")
                print(f"Total sequence length: {sequences.shape[1]}")
                print(f"Individual lengths: {packed_seq_lens}")
                if current_batch_images:
                    print(f"Images per sample: {[len(img_list) for img_list in current_batch_images]}")
                
                print(f"\nPacked Batch Statistics:")
                print(f"- Sequence shape: {sequences.shape}")
                print(f"- Number of sequences: {len(packed_seq_lens)}")
                print(f"- Individual lengths: {packed_seq_lens}")
                print(f"- Total tokens: {sum(packed_seq_lens)}")
                
                samples = Samples(
                    # Basic required fields
                    sequences=sequences,
                    num_actions=response_lengths,
                    response_length=torch.tensor(response_lengths, device="cuda", dtype=torch.float),
                    total_length=torch.tensor(total_lengths, device="cuda", dtype=torch.float),
                    
                    # Optional fields
                    attention_mask=attention_mask,
                    action_mask=None,
                    packed_seq_lens=packed_seq_lens,
                    
                    # Additional metadata
                    references=current_batch_references if current_batch_references else None,
                    images=current_batch_images if current_batch_images else None
                )
                samples_list.append(samples)
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
