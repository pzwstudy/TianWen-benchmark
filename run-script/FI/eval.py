import json
from sacrebleu.metrics import BLEU
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

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
    """计算准确率"""
    correct_count = 0
    for g, a in tqdm(zip(golden_data, answer_data), total=len(answer_data), desc="Calculating Accuracy"):
        flag = a.get("choose_id", "").strip()  # 避免 KeyError
        if not flag or flag not in g.get("choose", {}):  # 确保 key 存在
            continue
        if g["label"] == g["choose"][flag]:
            correct_count += 1

    accuracy = correct_count / len(answer_data) * 100 if answer_data else 0
    return accuracy

def bleu(golden_data, answer_data):
    """计算 BLEU 分数"""
    bleu = BLEU(tokenize="zh")
    total_bleu, count = 0.0, 0
    for g, a in tqdm(zip(golden_data, answer_data), total=len(answer_data), desc="Calculating BLEU Score"):
        if a.get("choose_id") not in {"A", "B", "C", "D"}:  # 过滤无效答案
            continue  
        
        reference = g.get("ans", [""])[0]  # 参考答案
        ans = a.get("ans_qa", "")  # 预测答案
        if not reference or not ans:
            continue

        total_bleu += bleu.sentence_score(ans, [reference]).score
        count += 1

    return total_bleu / count if count > 0 else 0

def similarity(golden_data, answer_data, model):
    """计算相似度"""
    total_similarity, count = 0.0, 0
    for g, a in tqdm(zip(golden_data, answer_data), total=len(answer_data), desc="Calculating Similarity"):
        if a.get("choose_id") not in {"A", "B", "C", "D"}:
            continue  

        reference = g.get("ans", [""])[0]
        ans = a.get("ans_qa", "")
        if not reference or not ans:
            continue

        # 计算嵌入
        try:
            ref_emb = model.encode(reference, normalize_embeddings=True)
            ans_emb = model.encode(ans, normalize_embeddings=True)
            similarity_score = np.dot(ref_emb, ans_emb)
            total_similarity += similarity_score
            count += 1
        except Exception as e:
            print(f"Embedding Error: {e}")

    return total_similarity / count if count > 0 else 0

def main():
    """主函数，执行评估流程"""
    golden_path = r"C:\Users\HIT\Desktop\data_fina\wenhua\data.json"
    ans_path = r"C:\Users\HIT\Desktop\data_fina\wenhua\vllm-qwen2.5-7b_lora-wenhua.json"
    model_name = r"C:\Users\HIT\Desktop\data_fina\xiaobu-embedding-v2"

    print("Loading data...")
    golden_data = readjson(golden_path)
    answer_data = readjson(ans_path)

    print("Loading embedding model...")
    model = SentenceTransformer(model_name)

    accuracy = acc(golden_data, answer_data)
    avg_bleu = bleu(golden_data, answer_data)
    avg_similarity = similarity(golden_data, answer_data, model)

    results = {
        "accuracy": accuracy,
        "avg_bleu": avg_bleu,
        "avg_similarity": avg_similarity
    }

    output_path = ans_path.replace(".json", "_eval.json")
    save_json(results, output_path)

    print(f"评估完成，结果已保存至 {output_path}")

if __name__ == "__main__":
    main()
