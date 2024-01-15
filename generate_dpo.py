from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from dataclasses import dataclass, field
from typing import Dict, Optional
import json
import torch
from datasets import Dataset, load_dataset

def load_json(path):
    with open(script_args.train_path, "r") as f:
        data = json.load(f)
    return Dataset.from_dict(data)

class ScriptArguments:
    """
    The arguments for evaluation.
    """
    base_model_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model path"})
    adapter_model_path: Optional[str] = field(default=None, metadata={"help": "the adapter model path"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    val_path: Optional[str] = field(
        default=None,
    )
    merge_path: Optional[str] = field(default="merged_adapters", metadata={"help": "the path to save the merged model"})

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model = AutoModelForCausalLM.from_pretrained(script_args.base_model_path)
    if script_args.use_peft:
        assert script_args.adapter_model_path is not None, "Please provide an adapter model path"
        model = PeftModel.from_pretrained(model, script_args.adapter_model_path)

    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_path)

    model = model.merge_and_unload()
    if script_args.merge_path is not None:
        model.save_pretrained(script_args.merge_path)
    
    # eval
    eval_dataset = load_json(script_args.val_path)
