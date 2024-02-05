from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, GenerationConfig
from dataclasses import dataclass, field
from typing import Dict, Optional
import json
import torch
from datasets import Dataset, load_dataset
import jsonlines
import numpy as np
import transformers
from dpo import get_summarize

def load_json(path):
    with open(script_args.train_path, "r") as f:
        data = json.load(f)
    return Dataset.from_dict(data)


@dataclass
class ScriptArguments:
    """
    The arguments for evaluation.
    """
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model path"})
    adapter_model_path: Optional[str] = field(default=None, metadata={"help": "the adapter model path"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    val_path: Optional[str] = field(
        default=None,
    )
    save_path: Optional[str] = field(
        default='eval_results.jsonl',
    )
    merge_path: Optional[str] = field(default=None, metadata={"help": "the path to save the merged model"})
    model_max_length: int = field(
    default=128,
    metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    inference_dtype: torch.dtype = field(
    default=torch.float32,
    metadata={"help": "The dtype to use for inference."},
    )



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=script_args.inference_dtype, device_map="auto",)
    if script_args.use_peft:
        assert script_args.adapter_model_path is not None, "Please provide an adapter model path"
        model = PeftModel.from_pretrained(model, script_args.adapter_model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=script_args.model_max_length,
    )
    responses = []
    data_id = []
    with jsonlines.open(script_args.val_path) as reader:
        for obj in reader:
            print(obj)
            inputs = tokenizer(obj['instruction'], return_tensors="pt", max_length=512)
            outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
                                    generation_config=generation_config,
                                    return_dict_in_generate=True,
                                    output_scores=True)
            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]

            response = tokenizer.decode(generated_tokens[0])
            obj["model_responses"] = response
            print("Response:", response)
            print()
            responses.append(obj)
    with jsonlines.open(script_args.save_path, mode='w') as writer:
        writer.write_all(responses)

