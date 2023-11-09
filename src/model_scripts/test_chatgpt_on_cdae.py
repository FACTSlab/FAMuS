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
import tiktoken

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

def generate_chat_gpt_response(prompt, 
                               openai: openai.api_resources,
                               model="gpt-3.5-turbo-0301",
                               max_response_tokens = 128,
                               temperature=0):
    
    num_tokens_in_prompt = num_tokens_from_messages(prompt, model=model)
    if num_tokens_in_prompt > (4096 - 128):
        max_response_tokens = 4096 - num_tokens_in_prompt

    response = openai.ChatCompletion.create(
        model=model,
        messages=prompt,
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
                        default="../../data/cross_doc_role_extraction/llm_prompt_format"
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        default="../../models/cross_doc_role_extraction/cdae/chatgpt/"
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

    ############################
    ## Report Data
    ############################
    with open(os.path.join(args.input_dir, 
                           "report_data",
                           f"{args.split}.json"), "r") as f:
        report_test_instances = [json.loads(line) for line in f.readlines()]

    # Run a loop over entire dataframe to get gpt responses
    # and save the responses to a file
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = "gpt-3.5-turbo-0301"
    generate_responses_with_retry(report_test_instances,
                                openai,
                                model_name = model_name,
                                max_response_tokens = 128,
                                temperature = 0,
                                output_path = args.output_dir,
                                output_gpt_filename = f"{args.split}_report_{model_name}_responses.jsonl"
                                    )
    
    ############################
    ## Source Data
    ############################
    with open(os.path.join(args.input_dir,
                            "source_data",
                            f"{args.split}.json"), "r") as f:
          source_test_instances = [json.loads(line) for line in f.readlines()]

    # Run a loop over entire dataframe to get gpt responses
    # and save the responses to a file
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = "gpt-3.5-turbo-0301"
    generate_responses_with_retry(source_test_instances,
                                openai,
                                model_name = model_name,
                                max_response_tokens = 128,
                                temperature = 0,
                                output_path = args.output_dir,
                                output_gpt_filename = f"{args.split}_source_{model_name}_responses.jsonl"
                                    )


if __name__ == "__main__":
    main()




