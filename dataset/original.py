"""最原始数据集构建，包括原文和摘要"""
import json
from utils.split import split

# 读取train.json文件
with open('D:\\毕业设计\\train.json', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 准备存储结果的列表
results = []

# 遍历每一行
for line in lines:
    item = json.loads(line)
    sentences = item["text"]
    input_text = "\n".join([sentence["sentence"] for sentence in sentences])
    output_text = item["summary"]

    # 将instruction和output作为字典存储
    result = {
        "instruction": input_text,
        "input": "",
        "output": output_text
    }

    results.append(result)

train_set, val_set, test_set = split(results)

# 将结果写入到新的JSON文件中
with open('D:\\毕业设计\\data\\dataset\\original\\train.json', 'w', encoding='utf-8') as f:
    json.dump(train_set, f, ensure_ascii=False, indent=4)

with open('D:\\毕业设计\\data\\dataset\\original\\valid.json', 'w', encoding='utf-8') as f:
    json.dump(val_set, f, ensure_ascii=False, indent=4)

with open('D:\\毕业设计\\data\\dataset\\original\\test.json', 'w', encoding='utf-8') as f:
    json.dump(test_set, f, ensure_ascii=False, indent=4)

print("数据集分割完成，训练集、验证集和测试集已分别保存到train.json、valid.json和test.json文件中。")
