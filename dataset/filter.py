import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat-4bits", trust_remote_code=True)

# 读取原始数据
with open('D:\\毕业设计\\data\\dataset\\cot\\test.json', 'r', encoding='utf-8') as file:
    data_list = json.load(file)


# 筛选函数
def filter_instructions(data):
    return [item for item in data if tokenizer(item["instruction"], return_length=True)["length"] < 2048]


# 应用筛选函数
filtered_data = filter_instructions(data_list)

# 将筛选后的数据写入新的JSON文件
with open('D:\\毕业设计\\data\\dataset\\cot\\filtered_test.json', 'w', encoding='utf-8') as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=4)

print("筛选完成，并已保存至 filtered_test.json")
