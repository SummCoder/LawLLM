import json
import random

# 用以将原有数据集划分为训练集、验证集以及测试集，比例为8：1：1


# 读取JSON文件
def read_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line:{e}")
    return data


# 划分数据集
def split_dataset(data, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
    # 确保比例之和为1
    assert train_ratio + validation_ratio + test_ratio == 1

    # 数据混洗
    random.shuffle(data)

    # 计算各数据集的大小
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    validation_size = int(total_size * validation_ratio)

    # 划分数据集
    train_set = data[:train_size]
    validation_set = data[train_size:train_size + validation_size]
    test_set = data[train_size + validation_size:]

    return train_set, validation_set, test_set


# 保存数据集，每条数据写入一行
def save_dataset(dataset, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        for item in dataset:
            json_str = json.dumps(item, ensure_ascii=False)
            file.write(json_str + '\n')


# 主函数
def main():
    # JSON文件路径
    file_path = 'D:\\毕业设计\\train.json'

    # 读取数据
    data = read_json_file(file_path)

    # 划分数据集
    train_set, validation_set, test_set = split_dataset(data)

    # 保存训练集、验证集和测试集
    save_dataset(train_set, '../data/train.json')
    save_dataset(validation_set, '../data/valid.json')
    save_dataset(test_set, '../data/test.json')


if __name__ == '__main__':
    main()
