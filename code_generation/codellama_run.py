from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
from time import time
import json
from torch.utils.data import Dataset, DataLoader
import argparse


def find_code_snippet(decoded_outout):
    code=''
    first_python_start = decoded_outout.find("[PYTHON]")
    second_python_start = decoded_outout.find("[PYTHON]", first_python_start + 1)
    third_python_start = decoded_outout.find("[PYTHON]", second_python_start + 1)
    if third_python_start != -1:
        third_python_end = decoded_outout.find("[/PYTHON]", third_python_start)
        stopword = decoded_outout.find("\ndef", third_python_start+10)
        stop_min = min(third_python_end,stopword)
        stop_max = max(third_python_end,stopword)
        if stop_min != -1:
            code = decoded_outout[third_python_start + len("[PYTHON]"):stop_min]
        elif stop_max != -1:
            code = decoded_outout[third_python_start + len("[PYTHON]"):stop_max]
        else:
            code = decoded_outout[third_python_start + len("[PYTHON]"):]
    return code.strip()

def fewshot_examples():
    """Loads and returns the few-shot examples for the task if they exist."""
    with open(
        "bigcode_eval/tasks/few_shot_examples/codesearchnet_few_shot_prompts.json", "r"
    ) as file:
        examples = json.load(file)
    return examples

def two_shot_prompt(entry, text, examples):
    """Two shot prompt format as instructions & solutions"""
    prompt = f"\nInstruction:\n{examples['instruction1']}\
                \nSolution:\n{examples['solution1']}\
                \nInstruction:\n{examples['instruction2']}\
                \nSolution:\n{examples['solution2']}\
                \nInstruction:\n{text}\
                \nSolution:\n"
    assert (
        prompt.count("Solution:\n") == 3
    ), "Splitting operation in postprocess_generation is invalid"
    return entry + prompt

def get_prompt(doc):
    """Builds the prompt for the LM to generate from."""
    examples = fewshot_examples()
    text = doc["nl"].split("concode_field_sep")[0].strip()
    if text.endswith("."):
        text = text[:-1].strip()
    entry = "Answer the following instructions in a one line of python code:\n"
    prompt = two_shot_prompt(entry, text, examples)
    return prompt

class CodeSearchNet(Dataset):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    def __init__(self, dataset_path):
        idx=0
        self.examples = []
        with open(dataset_path) as f:
            for line in f:
                idx+=1
                if idx % 3 == 0 or idx % 3 == 2:
                    continue
                line=line.strip()
                js=json.loads(line)
                self.examples.append({'docstring':" ".join(js['docstring_tokens']),'function':js['function']})
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return self.examples[i]['docstring'], self.examples[i]['function']

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='code_generation/codellama_7b_python', type=str,
                        help="The input path of the model checkpoint dir")
    parser.add_argument("--dataset", default='prompt_reformulation/dataset/test.jsonl', type=str,
                        help="The input path of the code search dataset")
    parser.add_argument("--output", default='code_generation/results/reform_test_pred.txt', type=str,
                        help="The output path of result file")
    args = parser.parse_args()

    model_dir = args.model
    DATASET_PATH = args.dataset
    output_dir = args.output

    instruction = '''[INST] Your task is to write a python function to solve a programming problem. 
    The Python code must be between [PYTHON] and [/PYTHON] tags.
    Generate code only (function does not include docstring).

    Problem: Get a JSON field from the response JSON
    [/INST]
    [PYTHON]
    def get_json_field(self, response_json, field_name):
        if field_name not in response_json:
            raise KeyError('Unable to get value for "%s" from Marathon '
                            'response: "%s"' % (
                            field_name, json.dumps(response_json),))
        return response_json[field_name]
    [/PYTHON]
            
    '''
    
    time_1 = time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16
    )
    model.to('cuda')
    print(f"Tokenizer & pipeline: {round(time() - time_1)} sec.")
    
    csn_data = CodeSearchNet(DATASET_PATH)
    data = DataLoader(csn_data, batch_size=1)

    time_1 = time()
    for idx, (nl, code) in enumerate(data):
        prompt = instruction+'[INST] Problem: ' + nl[0] + '\n[/INST]'
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            inputs["input_ids"].to('cuda'),
            max_new_tokens=200,
            do_sample=False,
            # top_p=0.9,
            # temperature=0.1,
        )
        output = output[0].to("cpu")
        decoded_output = tokenizer.decode(output)
        code = find_code_snippet(decoded_output)
        
        with open(output_dir, "a") as file:
            file.write(code.replace("\n", r"\n")+'\n')
        
        if idx % 100 == 99:
            print('{0} examples have been finished in {1} sec.'.format(idx+1, time()-time_1))

if __name__ == "__main__":
    main()