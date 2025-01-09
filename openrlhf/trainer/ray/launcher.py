import logging
import os
import socket
from typing import Callable, Dict, List, Optional, Type

import ray
import torch
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils.deepspeed import DeepspeedStrategy

from openrlhf.trainer.ray.utils import ray_noset_visible_devices

from qwen_vl_utils import process_vision_info

class DistributedTorchRayActor:
    def __init__(self, world_size, rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
        # environment variable for each actor, unless
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set, so
        # set local rank to 0 when the flag is not applicable.
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices() else "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BasePPORole(DistributedTorchRayActor):
    def _setup_distributed(self, strategy: DeepspeedStrategy):
        # configure strategy
        self.strategy = strategy
        strategy.setup_distributed()

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()


@ray.remote(num_gpus=1)
class ReferenceModelRayActor(BasePPORole):
    @ray.method(num_returns=1)
    def get_model_type(self):
        """Return the model type (e.g. 'causal_lm' or 'qwen2_vl')."""
        return self.model_type

    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        self.model_type = strategy.args.model_type  # Store model type
        model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            packing_samples=strategy.args.packing_samples,
            model_type=strategy.args.model_type,
        )
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        ring_attn_group=None,
        packed_seq_lens: Optional[list[int]] = None,
        images: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Forward pass for reference model."""
        device = torch.cuda.current_device()
        
        print(f"\n=== Reference Model Forward ===")
        print(f"Batch size: {sequences.size(0)}")
        print(f"Sequence length: {sequences.size(1)}")
        
        # Move inputs to device
        sequences = sequences.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            
        with torch.no_grad():
            try:
                # For Qwen2-VL processing
                if images and hasattr(self.model, 'processor'):
                    # Validate image count
                    if packed_seq_lens is not None:
                        assert len(images) == len(packed_seq_lens), \
                            f"Images count ({len(images)}) must match packed sequences count ({len(packed_seq_lens)})"
                    else:
                        assert len(images) == sequences.size(0), \
                            f"Images count ({len(images)}) must match batch size ({sequences.size(0)})"
                    
                    print("\n=== Image Processing ===")
                    print(f"Images count: {len(images)}")
                    print(f"First few images: {images[:2]}")
                    
                    # Create batch messages
                    batch_messages = []
                    if packed_seq_lens is not None:
                        offset = 0
                        for i, length in enumerate(packed_seq_lens):
                            sequence = sequences[0, offset:offset + length]
                            message = {"role": "user", "content": []}
                            
                            # Add image(s)
                            img_paths = images[i] if isinstance(images[i], list) else [images[i]]
                            for img_path in img_paths:
                                message["content"].append({
                                    "type": "image",
                                    "image": img_path
                                })
                            
                            # Add text
                            text = self.model.processor.decode(sequence)
                            message["content"].append({
                                "type": "text",
                                "text": text
                            })
                            batch_messages.append(message)
                            offset += length
                    else:
                        for i in range(sequences.size(0)):
                            message = {"role": "user", "content": []}
                            
                            # Add image(s)
                            img_paths = images[i] if isinstance(images[i], list) else [images[i]]
                            for img_path in img_paths:
                                message["content"].append({
                                    "type": "image",
                                    "image": img_path
                                })
                            
                            # Add text
                            text = self.model.processor.decode(sequences[i])
                            message["content"].append({
                                "type": "text",
                                "text": text
                            })
                            batch_messages.append(message)
                    
                    print(f"\n=== Batch Processing ===")
                    print(f"Processing {len(batch_messages)} messages")
                    print(f"Message structure:")
                    for i, msg in enumerate(batch_messages[:2]):
                        print(f"Message {i}:")
                        content = msg["content"]
                        print(f"- Content types: {[c['type'] for c in content]}")
                        print(f"- Images: {sum(1 for c in content if c['type'] == 'image')}")
                    
                    texts = [
                        self.model.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
                        for msg in batch_messages
                    ]
                    
                    # Process vision info with validation
                    print("\n=== Validating Image Inputs ===")
                    print(f"Batch size: {len(batch_messages)}")
                    print(f"Images count: {len(images)}")
                    
                    # Ensure image paths exist
                    valid_images = []
                    for img_list in images:
                        valid_img_list = []
                        for img_path in img_list:
                            if isinstance(img_path, str) and os.path.exists(img_path):
                                valid_img_list.append(img_path)
                            else:
                                print(f"Warning: Invalid image path - {img_path}")
                        valid_images.append(valid_img_list)
                    
                    print(f"Valid images count: {sum(len(img_list) for img_list in valid_images)}")
                    
                    # Process vision info
                    image_inputs, video_inputs = process_vision_info(batch_messages)
                    
                    # Validate image grid dimensions
                    if hasattr(image_inputs, 'shape'):
                        print(f"Image inputs shape: {image_inputs.shape}")
                        if image_inputs.shape[0] != len(batch_messages):
                            print(f"Adjusting image inputs from {image_inputs.shape[0]} to {len(batch_messages)}")
                            image_inputs = image_inputs[:len(batch_messages)]
                    
                    # Generate inputs through processor
                    print("\n=== Processor Inputs ===")
                    print(f"Texts count: {len(texts)}")
                    print(f"Images count: {len(image_inputs) if image_inputs is not None else 0}")
                    print(f"Videos count: {len(video_inputs) if video_inputs is not None else 0}")
                    
                    model_inputs = self.model.processor(
                        text=texts,
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Validate model inputs
                    print("\n=== Model Inputs Validation ===")
                    for key, value in model_inputs.items():
                        if isinstance(value, torch.Tensor):
                            print(f"{key}: shape {value.shape}")
                        elif isinstance(value, list):
                            print(f"{key}: length {len(value)}")
                    
                    # Validate image grid size
                    if 'image_grid_thw' in model_inputs:
                        grid_size = model_inputs['image_grid_thw'].size(0) if isinstance(model_inputs['image_grid_thw'], torch.Tensor) else len(model_inputs['image_grid_thw'])
                        batch_size = sequences.size(0)
                        if grid_size != batch_size:
                            # Adjust image grid size to match batch size
                            if isinstance(model_inputs['image_grid_thw'], torch.Tensor):
                                model_inputs['image_grid_thw'] = model_inputs['image_grid_thw'][:batch_size]
                            else:
                                model_inputs['image_grid_thw'] = model_inputs['image_grid_thw'][:batch_size]
                    
                    # Get model output
                    output = self.model(**model_inputs)
                    log_probs = output["logits"].to(torch.float32)
                else:
                    # Regular forward pass for non-VL models
                    output = self.model(
                        sequences,
                        attention_mask=attention_mask,
                        return_output=return_output,
                        ring_attn_group=ring_attn_group,
                        packed_seq_lens=packed_seq_lens,
                    )
                    log_probs = output
                
                print("\nForward pass completed successfully")
                return log_probs.to("cpu")
            
            except Exception as e:
                print(f"\nError in forward pass: {str(e)}")
                print(f"Sequences shape: {sequences.shape}")
                print(f"Images count: {len(images) if images else 0}")
                if images:
                    print(f"First few images: {images[:2]}")
                raise

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


@ray.remote(num_gpus=1)
class RewardModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = get_llm_for_sequence_regression(
            pretrain,
            "reward",
            normalize_reward=strategy.args.normalize_reward,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            value_head_prefix=strategy.args.value_head_prefix,
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(model)
        strategy.print("reward normalization status: {}".format(strategy.args.normalize_reward))
        strategy.print("mean: {}, std {}".format(model.mean, model.std))

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None, packed_seq_lens=None
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            reward = self.model(sequences.to(device), attention_mask.to(device), packed_seq_lens=packed_seq_lens)
        return reward.to("cpu")

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


class PPORayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[BasePPORole]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[BasePPORole],
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [
                {"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node} for _ in range(self._num_nodes)
            ]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready())
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(world_size, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(world_size, 0, None, None)
        self._actor_handlers = [master_actor]

        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank // self._num_gpus_per_node,
                        ),
                    ).remote(world_size, rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(world_size, rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        """Init model from pretrained checkpoint.

        Returns:
            List: list of remote object refs.
        """
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_fit_actor_model(
        self,
        critic_model_group: "PPORayActorGroup",
        initial_model_group: "PPORayActorGroup",
        reward_model_groups: List["PPORayActorGroup"],
        remote_rm_urls: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List = None,
    ):
        """Train actor model.

        Args:
            critic_model_group (PPORayActorGroup): critic model group.
            initial_model_group (PPORayActorGroup): reference model group.
            reward_model_groups (PPORayActorGroup): reward model groups.
            remote_rm_urls: remote RM APIs.
            reward_fn: reward calculate function, must be specified if using multiple reward models.
            vllm_engines: vllm engines for text generation, if not specified, generate text by actor model directly.

        Returns:
            List: list of remote object refs.
        """
        assert (
            (remote_rm_urls and len(remote_rm_urls) == 1)
            or (reward_model_groups and len(reward_model_groups) == 1)
            or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        critic_actors = critic_model_group._actor_handlers if critic_model_group else None
        initial_actors = initial_model_group._actor_handlers

        refs = []
        # TODO(wuxibin): actor model choose critic/reward/initial model in a
        # round robin fashion, implement more efficient dispatching strategy.
        for i, actor in enumerate(self._actor_handlers):
            critic_actor = critic_actors[i % len(critic_actors)] if critic_actors else None
            initial_actor = initial_actors[i % len(initial_actors)]

            reward_actors = []
            if not remote_rm_urls:
                for reward_model_group in reward_model_groups:
                    actors = reward_model_group._actor_handlers
                    reward_actors.append(actors[i % len(actors)])

            refs.append(
                actor.fit.remote(
                    critic_model=critic_actor,
                    initial_model=initial_actor,
                    reward_model=reward_actors,
                    remote_rm_url=remote_rm_urls,
                    reward_fn=reward_fn,
                    vllm_engines=vllm_engines,
                    # whether this actor should triger corresponding critic model training
                    critic_train_remote=(i < len(critic_actors)) if critic_actor else None,
                )
            )

        return refs

    def async_save_model(self):
        """Save actor model on rank 0.

        Returns:
            List: list of remote object refs.
        """
        return [actor.save_model.remote() for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs
