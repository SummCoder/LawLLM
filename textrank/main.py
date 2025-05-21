import json
import jieba
from TextRank import textRank
from rouge import Rouge
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 读取数据
with open('D:\\毕业设计\\data\\dataset\\original\\test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

rouge = Rouge()

jieba.load_userdict("D:\\毕业设计\\LawLLM\\textrank\\TextRank\\statics\\user_dict.txt")

total_bleu_score = 0
total_rouge_1 = 0
total_rouge_2 = 0
total_rouge_l = 0
count = 0

for item in tqdm(data, desc="Processing"):
    count += 1
    if count != 29:
        continue
    text = item["instruction"]
    reference = item["output"]

    # 调试输出，确认文本不为空
    if not text.strip() or not reference.strip():
        print("Warning: Empty source or reference, skipping.")
        continue

    # 使用TextRank提取摘要句子
    T = textRank.TextRank(text, pr_config={'alpha': 0.85, 'max_iter': 100})
    sentences_with_values = T.get_n_sentences(6)

    # 如果没提取到句子，跳过
    if not sentences_with_values:
        print("Warning: No sentences extracted by TextRank, skipping.")
        continue

    summary_sentences = [sentence for sentence, value in sentences_with_values]

    # 用空格连接句子，确保分词合理
    summary = ''.join(summary_sentences).strip()

    print(summary)

    if not summary:
        print("Warning: Empty summary, skipping.")
        continue

    # 中文分词
    reference_tokens = ' '.join(jieba.cut(reference))
    summary_tokens = ' '.join(jieba.cut(summary))

    total_bleu_score += sentence_bleu(
        references=[list(reference)],
        hypothesis=list(summary),
        smoothing_function=SmoothingFunction().method3)

    # 计算ROUGE分数
    rouge_scores = rouge.get_scores(summary_tokens, reference_tokens)

    total_rouge_1 += rouge_scores[0]['rouge-1']['f']
    total_rouge_2 += rouge_scores[0]['rouge-2']['f']
    total_rouge_l += rouge_scores[0]['rouge-l']['f']
    count += 1

if count > 0:
    avg_bleu = total_bleu_score / count
    avg_rouge_1 = total_rouge_1 / count
    avg_rouge_2 = total_rouge_2 / count
    avg_rouge_l = total_rouge_l / count

    print(f"Processed {count} samples.")
    print(f"Average BLUE-4: {avg_bleu:.4f}")
    print(f"Average ROUGE-1: {avg_rouge_1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge_2:.4f}")
    print(f"Average ROUGE-L: {avg_rouge_l:.4f}")
else:
    print("No valid samples processed.")
