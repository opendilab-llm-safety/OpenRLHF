from torch.utils.data import Dataset
from tqdm import tqdm


from typing import Union, Tuple, Optional

def preprocess_data(data, input_template=None, input_key="input", image_key=None, apply_chat_template=None) -> Union[str, Tuple[str, Optional[str]]]:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    # Handle image path if provided
    image_path = data.get(image_key) if image_key else None
    return (prompt, image_path) if image_path else prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        image_key=None,  # New: Key for image paths in dataset
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.images = []  # New: Store image paths
        self.references = []
        reference_key = getattr(self.strategy.args, "reference_key", "reference")
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            result = preprocess_data(data, input_template, input_key, image_key, apply_chat_template)
            if isinstance(result, tuple):
                prompt, image_path = result
                self.prompts.append(prompt)
                self.images.append(image_path)
            else:
                self.prompts.append(result)
                self.images.append(None)
            self.references.append(data.get(reference_key, None))  # 读取reference

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        image = self.images[idx]
        reference = self.references[idx]
        
        if reference is not None:
            if image is not None:
                return (prompt, reference, image)
            return (prompt, reference)
        if image is not None:
            return (prompt, image)
        return prompt
