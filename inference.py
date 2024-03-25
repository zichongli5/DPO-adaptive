# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: you need to install transformers from main to run this script. See https://huggingface.co/docs/transformers/installation#install-from-source
# TODO: bump transformers version in requirements at next release.

# 0. imports
from dataclasses import dataclass, field
from typing import Dict, Optional
import json
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers import LlamaForCausalLM, LlamaTokenizer
from trl import create_reference_model
from dpo_trainer import DPOTrainer

from accelerate import Accelerator, dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    
    adaptive_temp: Optional[bool] = field(default=False, metadata={"help": ""})
    

    # training parameters
    model_name_or_path: Optional[str] = field(default="./tldr_sft/", metadata={"help": "the model name"})
    adapter_model_path: Optional[str] = field(default="./rm_output/adaptive_dpo_lr5e-5_rho0.2_beta0.1/checkpoint-5060/", metadata={"help": "the adapter model path"})
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "batch size per device"})

    max_length: Optional[int] = field(default=600, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    output_path: Optional[str] = field(
        default='./eval_output/',
    )

    rho: Optional[float] = field(default=0.5, metadata={"help": "the rho parameter for adaptive_temperature loss"})


def construct_summarize_prompt(prompt):
    subreddit = prompt['subreddit'] if prompt['subreddit'] else ''
    title = prompt['title'] if prompt['title'] else ''
    post = prompt['post'] if prompt['post'] else ''
    # return "SUBREDDIT: " + prompt['subreddit'] + " " + "TITLE: " + prompt['title'] + " " + "POST: " + prompt['post']
    return "POST: " + post + "\n\nTL;DR:"

def get_summarize(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = '~/.cache/huggingface/datasets') -> Dataset:
    print(f'Loading OpenAI summarization dataset ({split} split) from Huggingface...')
    dataset = load_dataset('openai/summarize_from_feedback', 'comparisons', split=split, cache_dir=cache_dir)
    print('done')
    # # select only the first 1000 samples
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        return {
            "prompt": construct_summarize_prompt(sample['info']),
            "chosen": sample["summaries"][sample["choice"]]['text'],
            "rejected": sample["summaries"][1 - sample["choice"]]['text'],
        }

    return dataset.map(split_prompt_and_responses)


def load_json(path):
    with open(script_args.train_path, "r") as f:
        data = json.load(f)
    return Dataset.from_dict(data)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto",)
    model_ref.eval()
    # model_ref = create_reference_model(model)
    
    model = PeftModel.from_pretrained(model_ref, script_args.adapter_model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(script_args.adapter_model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # train_dataset = load_json(script_args.train_path)
    # eval_dataset = load_json(script_args.val_path)
    train_dataset = get_summarize("train", sanity_check=script_args.sanity_check)
    eval_dataset = get_summarize("validation", sanity_check=script_args.sanity_check)
    # print(isinstance(train_dataset, torch.utils.data.IterableDataset))

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        remove_unused_columns=False,
        learning_rate=0.0,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10,  # match results in blog post
        eval_steps=500,
        output_dir=script_args.output_path,
        optim="rmsprop",
        bf16=True,
    )

    peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
        )


    if script_args.adaptive_temp:
        print("Using adaptive temperature !!!!!")
    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=False,
        peft_config=peft_config,
        loss_type="sigmoid" if not script_args.adaptive_temp else "adaptive_temp",
        rho=script_args.rho,
        # ddp_find_unused_parameters=False
        # precompute_ref_log_probs = True
    )
    # dpo_trainer.train()
    # 6. train
    dpo_trainer.evaluate()
    print(dpo_trainer.results)








