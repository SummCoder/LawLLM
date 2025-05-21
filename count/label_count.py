import json

count = 0
text = 0


def transform_data(data):
    global count
    global text
    for item in data:
        text += 1
        item = json.loads(item)
        sentences = item['text']
        for index, sentence in enumerate(sentences):
            # 如果标签为1，则添加句子下标
            if sentence['label'] == 1:
                count += 1

    print(f"平均抽取摘要句子数量：{count / text}")


# 读取 JSON 文件
with open('D:\\毕业设计\\train.json', 'r', encoding='utf-8') as file:
    train_data = file.readlines()

# 调用函数并获取转换后的数据
transform_data(train_data)
