"""评测大语言模型生成的效果"""

import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import jieba
from tqdm import tqdm


# 读取JSON文件
def read_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


# 计算BLEU分数
def calculate_bleu(reference, summary):
    return sentence_bleu(references=[list(reference)],
                         hypothesis=list(summary),
                         smoothing_function=SmoothingFunction().method3)


# 计算ROUGE分数
def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']


# 主函数
def main():
    jieba.load_userdict("user_dict.txt")
    file_path = 'D:\\毕业设计\\result\\BertSum\\result.jsonl'
    data = read_jsonl(file_path)

    total_bleu = 0
    total_rouge_1 = 0
    total_rouge_2 = 0
    total_rouge_l = 0
    count = 0

    for record in tqdm(data, desc="Processing"):
        reference = record['label']
        summary = record['predict']

        # 中文分词
        reference_tokens = ' '.join(jieba.cut(reference))
        summary_tokens = ' '.join(jieba.cut(summary))

        bleu_score = calculate_bleu(reference, summary)
        rouge_1, rouge_2, rouge_l = calculate_rouge(reference_tokens, summary_tokens)

        total_bleu += bleu_score
        total_rouge_1 += rouge_1
        total_rouge_2 += rouge_2
        total_rouge_l += rouge_l
        count += 1

    average_bleu = total_bleu / count
    average_rouge_1 = total_rouge_1 / count
    average_rouge_2 = total_rouge_2 / count
    average_rouge_l = total_rouge_l / count

    print(f'BLEU-4 Score: {average_bleu}')
    print(f'ROUGE-1 Score: {average_rouge_1}')
    print(f'ROUGE-2 Score: {average_rouge_2}')
    print(f'ROUGE-l Score: {average_rouge_l}')
    print(f'TOTAL: {0.2*average_rouge_1+0.4*average_rouge_2+0.4*average_rouge_l}')


if __name__ == '__main__':
    main()
