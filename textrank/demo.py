from rouge import Rouge

rouge = Rouge()

rouge_score = rouge.get_scores(' '.join(list('这篇文章内容是新颖的')), ' '.join(list('文章内容新颖')))

print(rouge_score[0])
