import jsonlines
import pandas as pd
import torch.nn as nn


def find_overlap(input_tokens, output_tokens):
    start, end = 0, -1

    while input_tokens[start] == output_tokens[start]:
        start += 1
        if start >= min(len(input_tokens), len(output_tokens)):
            return -1, -1

    while input_tokens[end] == output_tokens[end]:
        end -= 1
        if start < max(len(input_tokens), len(output_tokens)):
            return -1, -1

    return start, end


# print(find_overlap("hello", "hello_word"))
def getOpDict(path='../data/dict/compare_equation - Sheet1.csv'):
    data = pd.read_csv(path)
    op_lst = data['op_id'].to_list()
    op_name = data['op_name'].to_list()

    op_dict = dict(zip(op_lst, op_name))
    print(op_dict)

    return op_dict


def match(test_path, generate_path, dict_path, out_path):
    op_dict = getOpDict(dict_path)
    predict = []
    with open(generate_path) as file:
        line = file.readline()
        while line:
            predict.append(line)
            line = file.readline()

    op_generate_lst = []
    with open("../data/processed/simplify/shanghai/op.txt") as file:
        line = file.readline()
        while line:
            op_generate_lst.append(line.replace('\n',''))
            line = file.readline()

    assert len(op_generate_lst) == len(predict)

    input = []
    output = []
    predict_result = []
    op_id_lst = []
    op_name_lst = []
    with open(test_path, 'r') as f:
        count = 0
        for item in jsonlines.Reader(f):

            if count < len(predict):
                item['predict'] = predict[count]
            else:
                item['predict'] = "还没生成出来"

            try:
                op_id = int(op_generate_lst[count])
            except ValueError:
                op_id = -1

            count += 1
            # op_id = int(item['input'].split(':')[0].strip())
            try:
                op_name = op_dict[op_id]
            except:
                op_name = "op_name文档暂缺"

            op_id_lst.append(op_id)
            op_name_lst.append(op_name)
            input.append(item['input'])

            output.append(item['output'])
            predict_result.append(item['predict'])

    assert len(predict_result) == len(predict)
    data = pd.DataFrame()
    data['op_name'] = op_name_lst
    data['op_id'] = op_id_lst
    data['input'] = input
    data['output'] = output
    data['predict'] = predict_result

    data.to_csv(out_path)


def clean_predict(input_path, output_path):
    with open(input_path, "r") as fin, open(output_path, "w+") as fout:
        line = fin.readline()
        listFormula = []
        while line:
            listFormula.append(
                line.replace('<pad>', '').replace('</s>', '').replace('\\ begin{ equation}', '\\begin{equation}')
                    .replace('\\ begin{ cases}', '\\begin{cases}')
                    .replace('\\ end{ equation}', '\\end{equation}')
                    .replace('\\ end{ cases}', '\\end{cases}')
                    .replace('\\ ', '\\')
            )
            line = fin.readline()

        for item in listFormula:
            fout.write(item)


if __name__ == '__main__':
    # clean_predict("../data/processed/simplify/generate/generated_predictions (5).txt",
    #               "../data/processed/simplify/generate/generated_predictions_clean_simplify.txt")
    match(test_path="../data/processed/simplify/test.json",
          generate_path="../data/processed/simplify/shanghai/dev_generated_predictions.txt"
          , dict_path='../data/dict/simplify_dict.csv',
          out_path="../data/processed/simplify/shanghai/t5_reverse_simplify_2w.csv")
