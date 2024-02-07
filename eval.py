## This code computes completions for questions in a json file using Claude-v2 model from Bedrock
## Follow instructions here for setup before running this code:
## https://quip-amazon.com/Lvo3A3JhGftQ/Using-Claude-from-AWS-Bedrock-Endpoint

import boto3
import json
import argparse
import os 
import datetime
import time
from tqdm import tqdm
import jsonlines
import numpy as np


def call_claude(data, args):
# secret


def pairwise_eval(args, data_path="../test_data/test_all.jsonl"):
    outputs1 = []
    outputs2 = []
    data = []
    

    ties, wins, loses = 0,0,0
    with jsonlines.open(args.input_file1) as reader:
        for obj in reader:
            outputs1.append(obj)
    
    with jsonlines.open(args.input_file2) as reader:
        for obj in reader:
            outputs2.append(obj)
    
    with jsonlines.open(args.data_path) as reader:
        for obj in reader:
            data.append(obj)
    
    # data = data[:730]

    try:
        saved_idx = list(np.load(args.saved_idx).astype(int))
    except:
        saved_idx = []    
    annotated = []
    for i in range(len(outputs1)):

        if i in saved_idx:
            continue

        results = []
            
        instruction = data[i]['instruction']
        output1 = outputs1[i]['model_responses']
        output2 = outputs2[i]['model_responses']

        # Analyze with prompt 1 first
        with open("summarize_prompt.txt", "r") as f:
            text = f.read()

        text = text.replace("||instruction||", instruction)
        text = text.replace("||output_1||", output1)
        text = text.replace("||output_2||", output2)

        completion = call_claude(text, args)
        

        print(text, completion)

        if completion is None:
            result = -10
        elif "A" in completion[-5:]:
            result = 0
        elif "B" in completion[-5:]:
            result = 1
        else:
            result = -10
        results.append(result)

        # Analyze with prompt 2 first
        with open("basic_prompt.txt", "r") as f:
            text = f.read()
    
        text = text.replace("||instruction||", instruction)
        text = text.replace("||output_1||", output2)
        text = text.replace("||output_2||", output1)

        completion = call_claude(text, args)

        print(text, completion)
        if completion is None:
            result = -10
        elif "A" in completion[-5:]:
            result = 1
        elif "B" in completion[-5:]:
            result = 0
        else:
            result = -10
        results.append(result)
        if sum(results) == 1:
            print("TIE")
            ties += 1
        elif sum(results) == 0:
            wins += 1
        elif sum(results) == 2:
            loses += 1
        annotated.append({"instruction": instruction,
                            "output1": output1,
                            "output2": output2,
                            "results": results})

        print("--------------------------------------------")
        print(f"{i} / {len(outputs1)}, ties: {ties}, wins: {wins}, loses: {loses}")
        print("--------------------------------------------")
        
        if -10 not in results:
            saved_idx.append(i)
            np.save(args.saved_idx, saved_idx)
                
            with jsonlines.open(f"{args.output_file}.jsonl", mode='a') as writer:
                writer.write_all([annotated[-1]])

    


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file1", default="./tldr_instruction_test.jsonl")
    parser.add_argument("--input_file2", default="./tldr_instruction_test.jsonl")
    parser.add_argument("--output_file", default="./claude_eval")
    parser.add_argument("--saved_idx")
    parser.add_argument("--data_path", default="./tldr_instruction_test.jsonl")
    parser.add_argument("--max_tokens_to_sample", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=250)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--max_threads", type=int, default=2)
    args = parser.parse_args()

    pairwise_eval(args)
