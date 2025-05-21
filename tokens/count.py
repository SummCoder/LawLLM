import json
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载 tokenizer，信任远程代码
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan2-7B-Base", trust_remote_code=True)

# 取训练集 JSON 文件
with open('D:\\毕业设计\\data\\dataset\\extracted\\train.json', 'r', encoding='utf-8') as f:
    train_set = json.load(f)

# 统计 token 数量
token_counts = []
for item in train_set:
    instruction = item["instruction"]
    # 对 instruction 进行编码
    encoding = tokenizer(instruction, return_length=True)
    token_count = encoding["length"]
    token_counts.append(token_count)

# 计算平均 token 数量
average_tokens = sum(token_counts) / len(token_counts)

print(f"原文平均占用的 token 数量: {average_tokens}")

# 统计各个长度段的数据个数和占比
length_segments = {
    "0-1024": 0,
    "1024-2048": 0,
    "2048-3072": 0,
    "3072-4096": 0,
    "4096以上": 0
}

# length_segments = {
#     "0-512": 0,
#     "512-1024": 0,
#     "1024以上": 0
# }

# for token_count in token_counts:
#     if token_count <= 512:
#         length_segments["0-512"] += 1
#     elif token_count <= 1024:
#         length_segments["512-1024"] += 1
#     else:
#         length_segments["1024以上"] += 1


for token_count in token_counts:
    if token_count <= 1024:
        length_segments["0-1024"] += 1
    elif token_count <= 2048:
        length_segments["1024-2048"] += 1
    elif token_count <= 3072:
        length_segments["2048-3072"] += 1
    elif token_count <= 4096:
        length_segments["3072-4096"] += 1
    else:
        length_segments["4096以上"] += 1

# 计算总数据量
total = len(token_counts)

# 打印各个长度段的个数和占比
for segment, count in length_segments.items():
    percentage = (count / total) * 100
    print(f"{segment}: 个数 = {count}, 占比 = {percentage:.2f}%")

# 绘制柱状图
segments = list(length_segments.keys())
counts = list(length_segments.values())

plt.figure(figsize=(10, 6))
plt.bar(segments, counts, color='skyblue')
plt.xlabel('Token 范围')
plt.ylabel('数量')
plt.title('原文长度分布')
plt.xticks(rotation=45)
plt.tight_layout()

# 显示图表
plt.savefig('sfzy_cail_input1.png')
