import json
from tqdm import tqdm
import os
from openai import OpenAI

golden_path = r"C:\Users\HIT\Desktop\data_fina\moniti\moniti.json"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

client = OpenAI(api_key="")

def readjson(path):
    with open(path, encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def acc(golden_data, ans_data):
    total = 0
    correct = 0
    length = len(ans_data)

    for i in tqdm(range(length), desc="Calculating Accuracy"):
        if "answer" in ans_data[i] and len(ans_data[i]["answer"]) > 0 and ans_data[i]["answer"][0] != -1:
            total += 1
            if ans_data[i]["answer"][0] == golden_data[i]["questions"][0]["answer"]:
                correct += 1

    true_correct = correct / total if total > 0 else 0
    false_correct = correct / length if length > 0 else 0
    return true_correct, false_correct

def gpt_score(qus, golden, ans):
    prompt = f"""
    你是一位精通古诗词的专家，擅长评估回答的准确性和完整性。请根据输入的古诗词内容、问题、标注答案和生成答案进行评分。  

    ## 评分规则：
    - **6 分（优秀）**：内容全面，准确理解题目要求，回答完整无遗漏。分析深入，观点清晰，有理有据。表达清晰，逻辑严密，语言流畅，无明显歧义或错漏。答案高度匹配标准答案，几乎无缺漏或偏差，核心观点和分析方式基本一致。  
    - **5 分（良好）**：内容基本完整，涵盖主要信息，但可能略有遗漏。分析较深入，能进行较好的解释和分析，但论述深度稍有欠缺。表达较清晰，逻辑基本通顺，语言较为流畅。答案与标准答案较为接近，但可能遗漏部分关键点，或表达上稍有欠缺。  
    - **4 分（中等）**：内容有一定准确性，涉及题目核心内容，但关键点不够充分。分析一般，论据较为薄弱，表达基本清楚，但逻辑性一般。答案部分匹配标准答案，但缺少较多重要信息，或有轻微理解偏差，分析较浅显。  
    - **3 分（一般）**：内容部分正确，但缺乏关键点，或理解有偏差。分析较弱，仅做表面分析或论述缺乏条理。表达有欠缺，逻辑较混乱。答案和标准答案差距较大，可能抓住了一些正确点，但遗漏核心内容，或存在较明显的误解。  
    - **2 分（较差）**：内容较为片面，可能有明显偏差。分析缺乏，基本没有逻辑推理，仅做简单描述或复述。表达较混乱，语句不连贯。答案与标准答案存在较大偏差，可能只有少量正确点，或回答表面化，缺乏有效分析。  
    - **1 分（极差）**：内容错误或无关，回答与题目无关或基本错误。无分析，仅简单堆砌词语或含糊回答。表达困难，语言不通顺。答案与标准答案完全不匹配，可能是无关内容、完全错误的理解，或根本没有回答问题。  

    ## 评分要求：
    - 依据标注答案评估生成答案的**准确性、完整性和表达质量**。  
    - **仅返回 JSON 格式的评分结果，不要额外输出解释或分析。**  

    ## 输入：
    - **问题**：{qus}  
    - **标注答案**：{golden}  
    - **生成答案**：{ans}  

    ## 返回格式：
    {{
        "score": X
    }}
    （X 为 1-6 之间的整数）
    """

    for _ in range(3):  # 三次重试机制
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "你是一个打分专家，根据标注答案和生成答案进行评分。"},
                          {"role": "user", "content": prompt}],
                stream=False
            )
            raw_response = response.choices[0].message.content.strip()
            
            json_response = raw_response.replace("```json", "").replace("```", "").strip()
            json_response = json_response.replace(" ", "").replace("\n", "").replace("\t", "")
            json_response = json_response.replace("：", ":").replace("，", ",").replace("。", ".")
            return json.loads(json_response)
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}，返回数据: {raw_response}")
        except Exception as e:
            print(f"处理数据时发生错误: {e}")
    return 0  # 若三次都失败，则返回 None

def calc_gpt_score(golden_data, ans_data):
    scores = []
    length = len(ans_data)
    total = 0
    total_score = 0

    for i in tqdm(range(length), desc="Calculating GPT Score"):
        if "answer" in ans_data[i] and len(ans_data[i]["answer"]) > 1 and ans_data[i]["answer"][0] != -1:
            qus = golden_data[i]["questions"][1]["question"]
            golden = golden_data[i]["questions"][1]["answer"]
            ans = ans_data[i]["answer"][1]
            total += 1
            score_result = gpt_score(qus, golden, ans)
            if score_result:
                score = score_result["score"]
                total_score += score
                scores.append(score)
    return total_score / total if total > 0 else 0, scores

def main():
    ans_path = r"C:\Users\HIT\Desktop\data_fina\moniti\claude-3-5-sonnet-20241022-moniti.json"
    output_path = ans_path.replace(".json", "_eval.json")
    ans_data = readjson(ans_path)
    golden_data = readjson(golden_path)

    if len(ans_data) != len(golden_data):
        print("Error: ans_data and golden_data have different lengths.")
    else:
        true_correct, false_correct = acc(golden_data, ans_data)
        gpt_avg_score, gpt_scores = calc_gpt_score(golden_data, ans_data)

        result = {
            "accuracy": {
                "true_correct_percentage": true_correct,
                "false_correct_percentage": false_correct,
            },
            "gpt_scores": {
                "average_score": gpt_avg_score,
                "individual_scores": gpt_scores,
            }
        }
        save_json(result, output_path)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
