import json

# 读取test.json文件
with open('D:\\毕业设计\\data\\dataset\\unilm\\test.json', 'r', encoding='utf-8') as f_test:
    test_data = [json.loads(line) for line in f_test]

# 读取predict.json文件
with open('D:\\毕业设计\\result\\predict.json', 'r', encoding='utf-8') as f_predict:
    predict_data = [line.strip() for line in f_predict]

# 确保两个文件中的记录数量相同
assert len(test_data) == len(predict_data), "The number of records in test.json and predict.json must be the same."

# 组合记录
result_data = []
for test_record, predict_record in zip(test_data, predict_data):
    combined_record = {
        "predict": predict_record,
        "label": test_record["output"]
    }
    result_data.append(combined_record)

# 写入result.json文件
with open('result.jsonl', 'w', encoding='utf-8') as f_result:
    for record in result_data:
        f_result.write(json.dumps(record, ensure_ascii=False) + '\n')

print("Records have been successfully combined and written to result.jsonl.")
