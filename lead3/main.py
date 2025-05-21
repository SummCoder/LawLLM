# coding=utf-8

import json
import re
import jieba
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

input_path = "D:\\毕业设计\\train.json"


def get_summary(text):
    for i, _ in enumerate(text):
        sent_text = text[i]["sentence"]
        if re.search(r"诉讼请求：", sent_text):
            text0 = text[i]["sentence"]
            text1 = text[i + 1]["sentence"]
            text2 = text[i + 2]["sentence"]
            break
        else:
            text0 = text[11]["sentence"]
            text1 = text[12]["sentence"]
            text2 = text[13]["sentence"]
    result = text0 + text1 + text2
    return result


if __name__ == "__main__":
    with open(input_path, 'r', encoding="utf8") as f:
        lines = f.readlines()

    jieba.load_userdict("user_dict.txt")

    selected_lines = lines[:len(lines)]

    # 初始化Rouge
    rouge = Rouge()

    # 初始化BLEU平滑函数
    smoothing = SmoothingFunction()

    total_bleu_score = 0
    total_rouge_1 = 0
    total_rouge_2 = 0
    total_rouge_l = 0
    count = 0

    for line in tqdm(selected_lines, desc="Processing"):
        data = json.loads(line)
        text = data.get('text')  # "text": [{"sentence":"001"},{"sentence":"002"}]
        reference = data.get('summary')  # 原始输出在summary字段

        # 调试输出，确认文本不为空
        if not text or not reference:
            print("Warning: Empty source or reference, skipping.")
            continue

        summary = get_summary(text)  # your model predict

        # 中文分词
        reference_tokens = ' '.join(jieba.cut(reference))
        summary_tokens = ' '.join(jieba.cut(summary))

        # 计算BLEU分数
        bleu_score = sentence_bleu(
            references=[list(reference)],
            hypothesis=list(summary),
            smoothing_function=smoothing.method3)

        total_bleu_score += bleu_score

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
        print(f"Average BLEU-4: {avg_bleu:.4f}")
        print(f"Average ROUGE-1: {avg_rouge_1:.4f}")
        print(f"Average ROUGE-2: {avg_rouge_2:.4f}")
        print(f"Average ROUGE-L: {avg_rouge_l:.4f}")
    else:
        print("No valid samples processed.")
