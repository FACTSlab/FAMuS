# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire
import json

from llama import Llama, Dialog

class Batch:
    """
    All the dialogs split into batches of 4.
    """
    def __init__(self, json_info):
        self.json_info = json_info
        self.dialogs = self._get_dialogs()
        self.batch_size = 2
        self.current = 0
        self.batches = self._get_batches()

    def _get_dialogs(self):
        dialogs = []
        for i in range(len(self.json_info)):
            dialogs.append([self.json_info[i]['instance_id'], self.json_info[i]['llm_prompt']])
        return dialogs
    
    def _get_batches(self):
        batches = []
        for i in range(0, len(self.dialogs), self.batch_size):
            batches.append(self.dialogs[i:i+self.batch_size])
        return batches
    
    


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    input_file: str = '/scratch/amart50/FACTS/data/famus_final_llm_prompt_format/llm_prompt_format_cdae/report_data/dev.json',
    output_file: str = '/scratch/amart50/FACTS/data/results/dev_report_llama_13b_responses.jsonl',
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # dialogs come from /scratch/amart50/FACTS/data/llm_prompt_format/source_data/test.json, this isn't truly a json file each line of the file is a json object 

    json_info = []
    with open(input_file, 'r') as f:
        for line in f:
            json_info.append(json.loads(line))

    print(len(json_info))
    batch_maker = Batch(json_info)
    print(len(batch_maker.batches))

    output_file_pth = output_file
    output_file = open(output_file_pth, 'w')
    while batch_maker.current < len(batch_maker.batches):
        batch = batch_maker.batches[batch_maker.current]
        prompts = [dialog[1] for dialog in batch]
        json_file = {}
        json_file['instance_id'] = batch[0][0]
        results = generator.chat_completion(
            prompts,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        for result in results:
            # output_file.write(json.dumps(result['generation']) + '\n')
            json_file['response'] = result['generation']['content']
            output_file.write(json.dumps(json_file) + '\n')
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")
        print(f"Batch {batch_maker.current} complete")
        batch_maker.current += 1
    output_file.close()
        



    # results = generator.chat_completion(
    #     dialogs,  # type: ignore
    #     max_gen_len=max_gen_len,
    #     temperature=temperature,
    #     top_p=top_p,
    # )

    # for dialog, result in zip(dialogs, results):
    #     for msg in dialog:
    #         print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    #     print(
    #         f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    #     )
    #     print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
