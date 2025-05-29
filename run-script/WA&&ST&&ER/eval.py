import json
import numpy as np
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer

golden_path = r"C:\Users\HIT\Desktop\data_fina\lvshi\lueshi_ans.json"

def readjson(path):
    with open(path, encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def acc(golden_data, ans_data):
    total, correct = 0, 0
    for i in tqdm(range(len(ans_data)), desc="Calculating Accuracy"):
        if ans_data[i]["choose_id"] != '-1':
            total += 1
            if ans_data[i]["choose_id"] == golden_data[i]["choose_id"]:
                correct += 1
    true_correct = correct / total if total > 0 else 0
    false_correct = correct / len(ans_data) if len(ans_data) > 0 else 0
    return true_correct, false_correct

def bleu(golden_data, ans_data):
    bleu = BLEU(tokenize="zh")

    bleu_1_words, bleu_2_words, bleu_3_words, bleu_4_words = [], [], [], []
    bleu_1_sents, bleu_2_sents, bleu_3_sents, bleu_4_sents = [], [], [], []

    for i in tqdm(range(len(ans_data)), desc="Calculating BLEU Score"):
        if ans_data[i]["choose_id"] != '-1':
            # 处理 ans_qa_words
            words_ans = ans_data[i].get("ans_qa_words", {})
            words_gold = golden_data[i].get("ans_qa_words", {})

            if isinstance(words_ans, dict):
                words_ans = list(words_ans.values())
            elif not isinstance(words_ans, list):
                words_ans = []

            if isinstance(words_gold, dict):
                words_gold = list(words_gold.values())
            elif not isinstance(words_gold, list):
                words_gold = []

            # 处理 ans_qa_sents
            sents_ans = ans_data[i].get("ans_qa_sents", {})
            sents_gold = golden_data[i].get("ans_qa_sents", {})

            if isinstance(sents_ans, dict):
                sents_ans = list(sents_ans.values())
            elif not isinstance(sents_ans, list):
                sents_ans = []

            if isinstance(sents_gold, dict):
                sents_gold = list(sents_gold.values())
            elif not isinstance(sents_gold, list):
                sents_gold = []

            # 确保参考答案和预测答案非空
            if not words_ans or not words_gold:
                print(f"Skipping BLEU calculation for index {i} due to empty words_ans or words_gold")
                continue
            if not sents_ans or not sents_gold:
                print(f"Skipping BLEU calculation for index {i} due to empty sents_ans or sents_gold")
                continue

            # 计算 BLEU
            bleu_result_words = bleu.corpus_score(words_ans, [words_gold])
            bleu_result_sents = bleu.corpus_score(sents_ans, [sents_gold])

            bleu_1_words.append(bleu_result_words.precisions[0])  
            bleu_2_words.append(bleu_result_words.precisions[1])  
            bleu_3_words.append(bleu_result_words.precisions[2])  
            bleu_4_words.append(bleu_result_words.precisions[3])  

            bleu_1_sents.append(bleu_result_sents.precisions[0])  
            bleu_2_sents.append(bleu_result_sents.precisions[1])  
            bleu_3_sents.append(bleu_result_sents.precisions[2])  
            bleu_4_sents.append(bleu_result_sents.precisions[3])  

    # 计算 BLEU 平均值
    avg_bleu = lambda lst: np.mean(lst) if lst else 0

    return {
        "word": {
            "avg_bleu": avg_bleu([avg_bleu(bleu_1_words), avg_bleu(bleu_2_words), avg_bleu(bleu_3_words), avg_bleu(bleu_4_words)]),
            "avg_bleu_1": avg_bleu(bleu_1_words),
            "avg_bleu_2": avg_bleu(bleu_2_words),
            "avg_bleu_3": avg_bleu(bleu_3_words),
            "avg_bleu_4": avg_bleu(bleu_4_words)
        },
        "sentence": {
            "avg_bleu": avg_bleu([avg_bleu(bleu_1_sents), avg_bleu(bleu_2_sents), avg_bleu(bleu_3_sents), avg_bleu(bleu_4_sents)]),
            "avg_bleu_1": avg_bleu(bleu_1_sents),
            "avg_bleu_2": avg_bleu(bleu_2_sents),
            "avg_bleu_3": avg_bleu(bleu_3_sents),
            "avg_bleu_4": avg_bleu(bleu_4_sents)
        }
    }

def sim(golden_data, ans_data, model):
    total_sim_words, total_sim_sents, count = 0.0, 0.0, 0
    for i in tqdm(range(len(ans_data)), desc="Calculating Semantic Similarity"):
        if ans_data[i]["choose_id"] != '-1':
            words_ans = ans_data[i].get("ans_qa_words", [])
            words_gold = golden_data[i].get("ans_qa_words", [])
            sents_ans = ans_data[i].get("ans_qa_sents", [])
            sents_gold = golden_data[i].get("ans_qa_sents", [])

            # **确保 words_ans 和 words_gold 是字符串列表**
            if isinstance(words_ans, dict):
                words_ans = list(words_ans.values())
            if isinstance(words_gold, dict):
                words_gold = list(words_gold.values())

            if isinstance(sents_ans, dict):
                sents_ans = list(sents_ans.values())
            if isinstance(sents_gold, dict):
                sents_gold = list(sents_gold.values())

            words_ans = [str(x) for x in words_ans if isinstance(x, str) and x.strip()]
            words_gold = [str(x) for x in words_gold if isinstance(x, str) and x.strip()]
            sents_ans = [str(x) for x in sents_ans if isinstance(x, str) and x.strip()]
            sents_gold = [str(x) for x in sents_gold if isinstance(x, str) and x.strip()]

            # **如果 words_ans 为空，跳过计算**
            if not words_ans or not words_gold:
                print(f"Skipping index {i}: words_ans or words_gold is empty")
                continue
            if not sents_ans or not sents_gold:
                print(f"Skipping index {i}: sents_ans or sents_gold is empty")
                continue

            # **调试输出**
            print(f"Index {i} - words_ans: {words_ans}")
            print(f"Index {i} - words_gold: {words_gold}")

            emb_words_ans = model.encode(words_ans, normalize_embeddings=True)
            emb_words_gold = model.encode(words_gold, normalize_embeddings=True)
            emb_sents_ans = model.encode(sents_ans, normalize_embeddings=True)
            emb_sents_gold = model.encode(sents_gold, normalize_embeddings=True)

            min_len = min(len(emb_words_ans), len(emb_words_gold))
            min_len_sents = min(len(emb_sents_ans), len(emb_sents_gold))

            sim_words = np.mean(np.sum(emb_words_ans[:min_len] * emb_words_gold[:min_len], axis=1)) if min_len > 0 else 0
            sim_sents = np.mean(np.sum(emb_sents_ans[:min_len_sents] * emb_sents_gold[:min_len_sents], axis=1)) if min_len_sents > 0 else 0

            total_sim_words += sim_words
            total_sim_sents += sim_sents
            count += 1

    return total_sim_words / len(ans_data) if len(ans_data)else 0, total_sim_sents / len(ans_data) if len(ans_data) else 0

def main():
    ans_path = r"C:\Users\HIT\Desktop\data_fina\lvshi\gpt-4-0613-lvshi.json"
    model = SentenceTransformer(r"C:\Users\HIT\Desktop\data_fina\xiaobu-embedding-v2")
    output_path = ans_path.replace(".json", "_eval.json")
    ans_data = readjson(ans_path)
    golden_data = readjson(golden_path)

    true_correct, false_correct = acc(golden_data, ans_data)
    bleu_results = bleu(golden_data, ans_data)
    sim_words, sim_sents = sim(golden_data, ans_data, model)

    save_json({
        "accuracy": {"true_correct": true_correct, "false_correct": false_correct},
        "BLEU": bleu_results,
        "semantic_similarity": {"words": sim_words, "sentences": sim_sents}
    }, ans_path.replace(".json", "_eval.json"))
    print(f"Evaluation results saved to {output_path}")
if __name__ == "__main__":
    main()
