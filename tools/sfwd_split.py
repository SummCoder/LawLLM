"""针对复旦大学数据集进行司法问答部分数据的处理，注意修改相关路径"""

import json
from utils.transform import transform_data, split_dataset, save_dataset


# 读取JSONL文件
def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
    return data


# 主函数
def main():
    # JSONL文件路径
    file_path_pair = 'D:\\毕业设计\\data\\law_chinese\\DISC-Law-SFT-Pair-QA-released.jsonl'
    file_path_triplet = 'D:\\毕业设计\\data\\law_chinese\\DISC-Law-SFT-Triplet-QA-released.jsonl'

    # 读取数据
    data_pair = read_jsonl_file(file_path_pair)
    data_triplet = read_jsonl_file(file_path_triplet)

    # 合并数据集
    combined_data = data_pair + data_triplet

    # 转换数据格式
    transformed_data = transform_data(combined_data)

    # 划分数据集
    train_set, validation_set, test_set = split_dataset(transformed_data)

    # 保存划分后的数据集
    save_dataset(train_set, '../data/sfwd/train.json')
    save_dataset(validation_set, '../data/sfwd/valid.json')
    save_dataset(test_set, '../data/sfwd/test.json')


if __name__ == '__main__':
    main()
