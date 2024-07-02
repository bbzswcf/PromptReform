import parso
import argparse
import PythonParsoPaser
from parso.python.tree import Name

def replace_identifiers(code_string):
    grammar = parso.load_grammar()
    parsed_ast = grammar.parse(code_string)
    if len(list(parsed_ast.iter_funcdefs())) == 0:
        print('error')
        return 
    func_code = list(parsed_ast.iter_funcdefs())[0]
    identifier_name_to_nodes = PythonParsoPaser.get_all_identifers(func_code)

    # 获取所有identifier
    identifier_list = list(identifier_name_to_nodes.keys())

    replace_dict = dict(zip(identifier_list[:-1], identifier_list[1:]))
    replace_dict.update({identifier_list[-1]: identifier_list[0]})

    for identifier, replace_str in replace_dict.items():
        nodes = identifier_name_to_nodes[identifier]
        for node in nodes:
            node.value = replace_str
    
    new_code_string = parsed_ast.get_code()
    return new_code_string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='code_generation/results/test_pred2.txt', type=str,
                        help="The input path of the code search dataset")
    parser.add_argument("--output", default='code_generation/results/test_pred_2.txt', type=str,
                        help="The output path of result file")
    args = parser.parse_args()
    code_lines = []
    # Read code from file
    with open(args.input, 'r', encoding='utf-8') as file:
        code_lines = file.readlines()
        # Process each code line
        for line in code_lines:
            modified_line = replace_identifiers(line)
            code_lines.append(modified_line)

    with open(args.output, 'a') as fe:
        for line in code_lines:
            fe.write(line + '\n')

if __name__ == "__main__":
    main()
