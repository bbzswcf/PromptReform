from openai import OpenAI
import argparse
import numpy as np
import json
from scipy.special import entr
import torch.nn.functional as F

client = OpenAI()

def read_json(filename):
    data = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            fields = ['idx','docstring','docstring_summary','docstring_tokens','function_tokens','function','return_statement','parameters','identifier']
            new_dict = {key: value for key, value in js.items() if key in fields}
            data[new_dict['idx']] = new_dict
            # data[js['url']] = js
    return data

def reformulate(data, entr_filename):
    # description = "Convert each TIF to PNG. Return filenames of new PNGs."
    idx = 0
    with open(entr_filename, 'a') as fe:
        for key in data:
            sentence = ' '.join(data[key]['docstring_tokens'])            

            response = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[
                  {"role": "system",
                    "content": "You are a code generation assistant tasked with enhancing prompts for clearer and more specific instructions. \
                        The original function description is unclear. \
                        Expand on the basis of the original function descriptions to help the LLMs generate better code"},
                  {"role": "user",
                  "content": "origin description: " + sentence + "\nexpanded description:"}
                ],
              temperature=1,
              max_tokens=256,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )
            fe.write(json.dumps(response.choices[0].message.content) + '\n')
            fe.flush()
            idx += 1
            if idx % 100 == 0:
                print(idx + " prompts have been reformulated.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='./prompt_reformulation/dataset/test.jsonl', type=str,
                        help="The input path of the code search dataset")
    parser.add_argument("--output", default='./prompt_reformulation/dataset/chatgpt.jsonl', type=str,
                        help="The output path of result file")
    args = parser.parse_args()

    data = read_json(args.dataset)
    entr_filename = args.output

    reformulate(data, entr_filename)

if __name__ == "__main__":
    main()