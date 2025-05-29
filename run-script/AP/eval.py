import json
golden_path = r"C:\Users\HIT\Desktop\data_fina\tiankong\tiankong-data.json"
def readjson(path):
    """读取 JSON 文件"""
    with open(path, encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

def save_json(data, file_path):
    """保存 JSON 文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
def acc(golden_data, answer_data):
    golden = 0  # 统计 golden_data 中答案的总数量
    answer = 0  # 统计匹配正确的数量
    length = len(golden_data)
    
    for i in range(length):
        m = len(golden_data[i]['ans']) if 'ans' in golden_data[i] else 0
        n = len(answer_data[i]['ans']) if 'ans' in answer_data[i] else 0
        
        golden += m  # 统计 golden_data 中答案的总数
        
        if m == 0 or n == 0:
            print(f"Warning: Missing 'ans' field at index {i}, m={m}, n={n}")
            continue  # 继续下一轮循环

        mini = min(m, n)  # 计算可对比的最小长度，防止越界
        
        for j in range(mini):
            ans_gold = golden_data[i]['ans'][j]
            ans_pred = answer_data[i]['ans'][j]

            # 确保是字符串类型
            if isinstance(ans_gold, list):
                ans_gold = " ".join(ans_gold)  # 连接成字符串
            if isinstance(ans_pred, list):
                ans_pred = " ".join(ans_pred)

            if isinstance(ans_gold, str) and isinstance(ans_pred, str):
                if ans_gold.strip() == ans_pred.strip():
                    answer += 1
                    print(f"Matched: {ans_gold.strip()} == {ans_pred.strip()}")
            else:
                print(f"Warning: Non-string data at index [{i}][{j}], skipping comparison.")

    return answer / golden if golden > 0 else 0  # 避免 ZeroDivisionError


        
def main():
    # 读取 JSON 文件
    ans_path =r"C:\Users\HIT\Desktop\data_fina\tiankong\vllm-chatglm4-9b-tiankong.json"
    accuracy = 0.0
    output_path =ans_path.replace(".json", "_eval.json")
    golden_data = readjson(golden_path)
    answer_data = readjson(ans_path)
    if len(answer_data) != len(golden_data):
        print("Error: ans_data and golden_data have different lengths.")
    else:
        # 计算准确率
        accuracy = acc(golden_data,answer_data)
        # acc写入新的 JSON 文件
        result = {
            "accuracy": accuracy
        }
        save_json(result, output_path)
        print(f"Results saved to {output_path}")
if __name__ == "__main__":
    main()
    
