import random


def split(results):
    # 打乱数据
    random.shuffle(results)

    # 计算每个集合的大小
    total = len(results)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    test_size = total - train_size - val_size

    # 分割数据
    train_set = results[:train_size]
    val_set = results[train_size:train_size + val_size]
    test_set = results[train_size + val_size:]
    return train_set, val_set, test_set
