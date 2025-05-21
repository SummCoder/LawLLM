import random
import json


def transform_data(data):
    transformed_data = []
    for entry in data:
        # 假设每个entry都有input和output键
        new_entry = {
            "instruction": entry["input"],
            "input": "",
            "output": entry["output"]
        }
        transformed_data.append(new_entry)
    return transformed_data


def split_dataset(data, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
    assert train_ratio + validation_ratio + test_ratio == 1
    random.shuffle(data)
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    validation_size = int(total_size * validation_ratio)
    train_set = data[:train_size]
    validation_set = data[train_size:train_size + validation_size]
    test_set = data[train_size + validation_size:]
    return train_set, validation_set, test_set


# 保存数据集，每条数据写入一行
def save_dataset(dataset, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)
