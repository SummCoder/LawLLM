"""针对复旦大学数据集用以将原有数据集转化为进行指令监督微调数据集，划分为训练集、验证集以及测试集，比例为8：1：1"""

import json
from utils.transform import transform_data, split_dataset, save_dataset


# 读取JSON文件
def read_json_file(file_path, max_records=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if max_records is not None and i >= max_records:
                break
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
    return data


# 主函数
def main():
    # JSON文件路径
    file_path = 'D:\\毕业设计\\DISC-Law-SFT-Pair.jsonl'

    # 读取数据
    data = read_json_file(file_path, max_records=8234)

    # 转换数据格式
    transformed_data = transform_data(data)

    # 划分数据集
    train_set, validation_set, test_set = split_dataset(transformed_data)

    # 保存训练集、验证集和测试集
    save_dataset(train_set, '../data/sfzy/train.json')
    save_dataset(validation_set, '../data/sfzy/valid.json')
    save_dataset(test_set, '../data/sfzy/test.json')


if __name__ == '__main__':
    main()
