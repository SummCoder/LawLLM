import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 指标名称
labels = ['BLEU-4', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']

# 对应指标数值
score_temp_05 = [17.79, 37.02, 17.83, 28.04]
score_temp_095 = [16.89, 36.61, 17.27, 27.32]

x = np.arange(len(labels))  # 标签位置
width = 0.35  # 柱宽

fig, ax = plt.subplots(figsize=(8, 5))
rects2 = ax.bar(x - width/2, score_temp_095, width, label='温度=0.95', color='#ff7f0e')
rects1 = ax.bar(x + width/2, score_temp_05, width, label='温度=0.5', color='#1f77b4')


ax.set_ylabel('分数')
# ax.set_title('不同温度设置下的模型性能对比')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 40)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 给柱状图添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 文字上移3个点
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('temperature_comparison.png', dpi=300)  # 保存为文件
plt.show()



