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
        "output": output_text
    }

    results.append(result)

train_set, val_set, test_set = split(results)


# 定义一个函数，用于将数据集写入JSONL文件
def write_to_jsonl(dataset, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in dataset:
            # 将每个item转换为JSON字符串，并写入文件的一行
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# 将结果写入到新的JSONL文件中
write_to_jsonl(train_set, 'D:\\毕业设计\\data\\dataset\\unilm\\train.json')
write_to_jsonl(val_set, 'D:\\毕业设计\\data\\dataset\\unilm\\valid.json')
write_to_jsonl(test_set, 'D:\\毕业设计\\data\\dataset\\unilm\\test.json')

print("数据集分割完成，训练集、验证集和测试集已分别保存到train.json、valid.json和test.json文件中。")
