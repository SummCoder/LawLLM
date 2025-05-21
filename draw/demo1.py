import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建DataFrame
data = {
    '方法': ['Lead3', 'Textrank', 'BertSum', 'UniLM',
             'Baichuan2-7B-Base（微调前）', 'Baichuan2-7B-Base（微调后）',
             'Baichuan2-7B-Chat（微调前）', 'Baichuan2-7B-Chat（微调后）',
             'Baichuan2-7B-Chat（添加One-Shot模板）', 'Baichuan2-7B-Chat（使用思维链模板）'],
    'BLEU-4': [10.11, 20.87, 12.58, 13.08, 0.13, 53.74, 13.38, 54.39, 16.63, 15.64],
    'ROUGE-1': [31.36, 40.34, 30.67, 38.66, 1.55, 69.09, 31.58, 69.62, 34.65, 33.64],
    'ROUGE-2': [13.54, 22.43, 13.51, 20.89, 0.06, 53.37, 13.88, 53.80, 16.16, 15.32],
    'ROUGE-L': [21.98, 24.55, 19.77, 30.96, 0.90, 62.36, 21.98, 62.94, 24.82, 22.85]
}

df = pd.DataFrame(data)

# 设置绘图参数
methods = df['方法']
metrics = ['BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
x = np.arange(len(methods))
width = 0.2

fig, ax = plt.subplots(figsize=(16, 8))

for i, metric in enumerate(metrics):
    ax.bar(x + i * width, df[metric], width, label=metric)

ax.set_xlabel('方法')
ax.set_ylabel('分数')
ax.set_title('各方法的实验结果')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('result.png', dpi=300)  # 保存为文件
plt.show()