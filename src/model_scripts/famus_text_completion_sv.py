# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import json

from llama import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    input_file = '/scratch/amart50/FACTS/data/famus_final_llm_prompt_format/llm_prompt_format_sv/dev.jsonl',
    output_file = '/scratch/amart50/FACTS/data/results/dev_sv_llama_13b_responses.jsonl'
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    json_files = []
    with open(input_file, 'r') as f:
        for line in f:
            json_files.append(json.loads(line))
        
    output_file = open(output_file, 'w')
    for json_file in json_files:
        output_json = {}
        output_json['instance_id'] = json_file['instance_id']
        prompts = [json_file['llm_prompt']]
        print(json_file)
        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        output_json['llm_response'] = results[0]['generation']
        for prompt, result in zip(prompts, results):
            print(f"> {result['generation']}")
            print("\n==================================\n")
        output_file.write(json.dumps(output_json) + '\n')
    output_file.close()


if __name__ == "__main__":
    fire.Fire(main)
