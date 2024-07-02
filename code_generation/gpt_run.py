import argparse
import re
import json
from openai import OpenAI
from time import time

client = OpenAI()

def read_json(filename):
    data = []
    idx = 0
    with open(filename) as f:
        for line in f:
            idx+=1
            if idx % 3 == 1:
                line = line.strip()
                js = json.loads(line)
                data.append(js) 
    return data

def gpt_api_call(**kwargs):
    response = client.chat.completions.create(**kwargs)
    return response

def generate(data, entr_filename):
    pattern = re.compile(r'[\u200b-\u200f\uFEFF\u202a-\u202e]')
    idx = 0
    with open(entr_filename, 'a', encoding='utf-8') as fe:
        time_1 = time()
        for key in data:
            idx += 1
            response = gpt_api_call(model="gpt-3.5-turbo",
            messages = [{"role": "system","content": "You are an AI programming assistant \
                             that writes a python function to solve a programming problem. \
                             Your language is python. Don't explain the code. \
                             Don't ask any extra questions, just generate code block itself.\
                             Ensure that the code contains no import statements."},
                            {"role": "user","content": "Get a JSON field from the response JSON"},
                            {"role": "assistant","content": "def get_json_field(self, response_json, \
                             field_name):\n    if field_name not in response_json:\n        raise \
                             KeyError('Unable to get value for \"%s\" from Marathon '\n            \
                                         'response: \"%s\"' % (\n                        field_name, \
                             json.dumps(response_json),))\n    return response_json[field_name]"},
                            {"role": "user", "content": " ".join(key['docstring_tokens'])}],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
                        
            code = response.choices[0].message.content
            code = pattern.sub("", code)
            fe.write(code.replace("\n", r"\n").replace("\r", r"\r")+'\n')
            fe.flush()
            
            if idx % 100 == 0:
                print("{0} examples finished in {1}sec.".format(idx, time()-time_1))
        
    
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_file", default='prompt_reformulation/dataset/test.jsonl', type=str,
                        help="An optional input evaluation data file.")
    parser.add_argument("--output_dir", default='code_generation/results/reform_test_pred.txt', type=str, 
                        help="The output directory where the model predictions will be written.")
    
    args = parser.parse_args()

    data = read_json(args.data_file)

    generate(data, args.output_dir)


if __name__ == "__main__":
    main()