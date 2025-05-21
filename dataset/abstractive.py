"""构建多样化的输入，摘要作为输出"""
import json
import random
from utils.split import split

# 定义提示语句列表
prompts = [
    "请大致描述这篇文书的内容：\n",
    "请大致描述这篇文书的内容\n",
    "请对这篇法律文书进行摘要\n\n",
    "请归纳这篇文书的大致要点：\n\n",
    "\n以上是一篇法律文书，请归纳这篇文书的大致要点。",
    "\n请对其进行摘要。"
]

# 读取train.json文件
with open('D:\\毕业设计\\train.json', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 准备存储结果的列表
results = []

# 遍历每一行
for line in lines:
    item = json.loads(line)
    sentences = item["text"]
    # input_text = "\n".join([sentence["sentence"] for sentence in sentences if sentence["label"] == 1])
    input_text = "\n".join([sentence["sentence"] for sentence in sentences])
    output_text = item["summary"]

    # 随机选择一个提示语句
    prompt = random.choice(prompts)

    # 根据提示语句决定如何构建instruction
    if prompt == "\n以上是一篇法律文书，请归纳这篇文书的大致要点。" or prompt == "\n请对其进行摘要。":
        instruction = input_text + prompt
    else:
        instruction = prompt + input_text

    # 将instruction和output作为字典存储
    result = {
        "instruction": instruction,
        "input": "",
        "output": output_text
    }

    results.append(result)

train_set, val_set, test_set = split(results)

# 将结果写入到新的JSON文件中
with open('D:\\毕业设计\\data\\dataset\\abstract\\train.json', 'w', encoding='utf-8') as f:
    json.dump(train_set, f, ensure_ascii=False, indent=4)

with open('D:\\毕业设计\\data\\dataset\\abstract\\valid.json', 'w', encoding='utf-8') as f:
    json.dump(val_set, f, ensure_ascii=False, indent=4)

with open('D:\\毕业设计\\data\\dataset\\abstract\\test.json', 'w', encoding='utf-8') as f:
    json.dump(test_set, f, ensure_ascii=False, indent=4)

print("数据集分割完成，训练集、验证集和测试集已分别保存到train.json、valid.json和test.json文件中。")
