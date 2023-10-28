import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from tqdm import tqdm
import os
import json
import argparse

def generate_chat_gpt_response(prompt, 
                               openai: openai.api_resources,
                               model="gpt-3.5-turbo-0301",
                               max_response_tokens = 128,
                               temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens = max_response_tokens,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_chat_gpt_response_with_retry(prompt, 
                                          openai, 
                                          model_name, 
                                          max_response_tokens, 
                                          temperature):
    
    return generate_chat_gpt_response(prompt, 
                                      openai, 
                                      model_name, 
                                      max_response_tokens, 
                                      temperature)


def generate_responses_with_retry(test_instances,
                                  openai, 
                                  model_name,
                                  max_response_tokens, 
                                  temperature, 
                                  output_path, 
                                  output_gpt_filename):
    responses = []
    for instance in tqdm(test_instances):
        prompt = instance['llm_prompt']

        response = generate_chat_gpt_response_with_retry(prompt,
                                            openai,
                                            model_name = model_name,
                                            max_response_tokens = max_response_tokens,
                                            temperature=temperature)
        
        response_dict = {'instance_id': instance['instance_id'],
                         'response': response,
                         'prompt': prompt}
                         
        responses.append(response_dict)
        
        with open(os.path.join(output_path, output_gpt_filename), "a") as f:
            f.write(json.dumps(response_dict))
            f.write("\n")
    return responses

def parse_args():
    # input dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", 
                        type=str, 
                        default="../../data/source_validation/"
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        default="../../models/source_validation/chatgpt/"
                        )
    parser.add_argument("--split",
                        type=str,
                        default="test"
                        )
    args = parser.parse_args()
    return args
    

def main():
    # Load the llm prompt test instances
    args = parse_args()
    EMBEDDING_ENCODING = 'cl100k_base'
    with open(os.path.join(args.input_dir, f"{args.split}.jsonl"), "r") as f:
        test_instances = [json.loads(line) for line in f.readlines()]

    # Run a loop over entire dataframe to get gpt responses
    # and save the responses to a file
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = "gpt-3.5-turbo-0301"
    generate_responses_with_retry(test_instances,
                                openai,
                                model_name = model_name,
                                max_response_tokens = 128,
                                temperature = 0,
                                output_path = args.output_dir,
                                output_gpt_filename = f"{args.split}_{model_name}_responses.jsonl"
                                    )


if __name__ == "__main__":
    main()




