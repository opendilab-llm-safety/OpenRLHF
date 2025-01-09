import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
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
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    references: Optional[List[str]], the reference answers for each sample.
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    references: Optional[List[str]] = None
    images: Optional[List[str]] = None


class NaiveExperienceMaker(ABC):
    """Naive experience maker."""

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
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
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
    # Rest of the class implementation remains unchanged...
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

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str], Tuple[List[str], List[str], List[str]]], **generate_kwargs) -> List[Experience]:
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
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                references=batch_references,
                images=batch_images,
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

        print(f"===========samples: {samples}")
        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # Get images from samples if available
        images = samples.images if hasattr(samples, "images") else None

        # log probs
        print("\n=== Actor Forward ===")
        print(f"sequences: {sequences.shape}")
        print(f"num_actions: {num_actions}")
        print(f"attention_mask: {attention_mask.shape}")
        print(f"images: {images.shape if images is not None else 'None'}")
        action_log_probs = self.actor(sequences, num_actions, attention_mask, images=images)
        print(f"action_log_probs: {action_log_probs.shape}")
        

        # init log probs
        print("\n=== Initial Model Forward ===")
        print(f"sequences: {sequences.shape}")
        print(f"num_actions: {num_actions}")
        print(f"attention_mask: {attention_mask.shape}")
        print(f"images: {images.shape if images is not None else 'None'}")
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask, images=images)
        print(f"base_action_log_probs: {base_action_log_probs.shape}")

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask, images=images)
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            print("\n=== Reward Model Inputs ===")
            # 直接使用完整的sequences进行解码 
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            
            # 处理references
            references = None
            if hasattr(samples, "references") and samples.references is not None:
                references = samples.references
                print(f"References available: {len(references)}")
                print(f"First few references: {references[:2]}")
            else:
                print("No references available")
            
            print(f"Input sequence shape: {sequences.shape}")
            print(f"Action mask shape: {action_mask.shape}")
            print(f"Queries count: {len(queries)}")
            print(f"First query:\n{queries[0]}")
            print(f"Remote RM URL: {self.remote_rm_url}")
            
            r = remote_rm_fn(
                self.remote_rm_url, 
                queries=queries,
                references=references
            ).to(device=action_log_probs.device)
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
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
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
        
        print(f"===========samples: {samples}")
        # print(f"===========Processing samples with images: {samples.images}")
        # print(f"===========references: {samples.references}")
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # Get images from samples if available
        images = samples.images if hasattr(samples, "images") else None
        
        # log probs
        print("\n=== Actor Forward ===")
        print(f"Sequences shape: {sequences.shape}")
        print(f"Images count: {len(images) if images else 0}")
        action_log_probs = self.actor(
            sequences, 
            num_actions, 
            attention_mask, 
            packed_seq_lens=packed_seq_lens,
            images=images
        )
        print(f"action_log_probs: {action_log_probs.shape}")

        # init log probs
        # Check model types and prepare images
        initial_model_type = ray.get(self.initial_model.get_model_type.remote())
        images_arg = images if initial_model_type == "qwen2_vl" else None
        
        # init log probs
        print("\n=== Initial Model Forward ===")
        print(f"Sequences shape: {sequences_cpu.shape}")
        print(f"Images count: {len(images) if images else 0}")
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu,
            num_actions,
            attention_mask_cpu,
            images=images_arg,
            return_output=False,
            ring_attn_group=None,
            packed_seq_lens=packed_seq_lens
        )
        print(f"base_action_log_probs_ref: {base_action_log_probs_ref}")

        # values
        if self.critic is not None:
            # Check if critic supports vision
            critic_supports_vision = ray.get(self.critic.has_vision_support.remote())
            
            print(f"\n=== Critic Forward ===")
            print(f"Critic supports vision: {critic_supports_vision}")
            print(f"Sequences shape: {sequences_cpu.shape}")
            print(f"Images count: {len(images) if images else 0}")
            if images:
                print(f"First few images: {images[:2]}")
            
            # Forward with vision support check
            value_ref = self.critic.forward.remote(
                sequences_cpu,
                num_actions,
                attention_mask_cpu,
                images=images if critic_supports_vision else None,
                return_output=False,
                ring_attn_group=None,
                packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
        else:
            # remote RM
            if not self.packing_samples:
                queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
            else:
                sequences_list = []
                offset = 0
                tokens_list = sequences_cpu.tolist()[0]
                for length in packed_seq_lens:
                    sequences_list.append(tokens_list[offset : offset + length])
                    offset += length
                queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

            references = None
            if hasattr(samples, "references") and samples.references is not None:
                references = samples.references

            for rm in self.remote_rm_url:
                print(f"Remote RM: {rm}")
                print(f"Queries: {queries}")
                print(f"References: {references}")
                r = remote_rm_fn_ray.remote(rm, queries=queries, references=references)
                r_refs.append(r)


        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
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
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        images=current_batch_images,
                        references=current_batch_references
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
                    sequences=sequences,
                    attention_mask=attention_mask,
                    action_mask=None,
                    num_actions=response_lengths,
                    packed_seq_lens=packed_seq_lens,
                    response_length=torch.tensor(response_lengths, device="cuda", dtype=torch.float),
                    total_length=torch.tensor(total_lengths, device="cuda", dtype=torch.float),
                    images=current_batch_images if current_batch_images else None,
                    references=current_batch_references if current_batch_references else None
                )
                samples_list.append(samples)
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
