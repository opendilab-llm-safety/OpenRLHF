from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
import sys
import os

# Add the model directory to Python path
model_path = "/mnt/hwfile/llm-safety/models/InternVL2_5-QwQ-38B-v5"
if model_path not in sys.path:
    sys.path.append(model_path)

from configuration_internvl_chat import InternVLChatConfig
from modeling_internvl_chat import InternVLChatModel

def register_model():
    # Register the configuration and model mappings
    AutoConfig.register("internvl_chat", InternVLChatConfig)
    
    # Update the model mappings
    AutoModel._model_mapping.register(InternVLChatConfig, InternVLChatModel)
    AutoModelForCausalLM._model_mapping.register(InternVLChatConfig, InternVLChatModel)
    
    # Update reverse mappings for configurations
    model_type = "internvl_chat"
    AutoModel._model_mapping._reverse_config_mapping[InternVLChatConfig.__name__] = model_type
    AutoModelForCausalLM._model_mapping._reverse_config_mapping[InternVLChatConfig.__name__] = model_type

if __name__ == "__main__":
    register_model()
    print("InternVL chat model and config registered successfully")
