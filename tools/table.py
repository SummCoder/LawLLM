import matplotlib.pyplot as plt

# 数据
categories_instruction = ['0-1024', '1024-2048', '2048-3072', '3072-4096', '4096以上']
counts_instruction = [975, 4740, 1487, 294, 91]
percentages_instruction = [12.85, 62.48, 19.60, 3.88, 1.20]

categories_summary = ['0-512', '512-1024', '1024以上']
counts_summary = [7587, 0, 0]
percentages_summary = [100.00, 0.00, 0.00]

# 创建柱状图 - 指令平均占用的 token 数量分布
plt.figure(figsize=(10, 5))
plt.bar(categories_instruction, counts_instruction, color='skyblue')
plt.xlabel('Token 数量区间')
plt.ylabel('指令数量')
plt.title('指令平均占用的 Token 数量分布')
plt.xticks(rotation=45)
for i, v in enumerate(counts_instruction):
    plt.text(i, v + 50, f'{percentages_instruction[i]}%', ha='center')

# 显示图表
plt.show()

# 创建饼图 - 生成式摘要平均占用的 token 数量占比
plt.figure(figsize=(7, 7))
plt.pie(counts_summary, labels=categories_summary, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightgrey', 'lightyellow'])
plt.title('生成式摘要平均占用的 Token 数量占比')

# 显示图表
plt.show()
