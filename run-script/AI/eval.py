import json
from tqdm import tqdm
import os
from openai import OpenAI

golden_path = r"C:\Users\HIT\Desktop\data_fina\diangu\diangu.json"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 配置 OpenAI API 客户端
client = OpenAI(
    api_key="",  # 替换为你的 API 密钥
)

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
    correct = 0
    length = len(golden_data)
    for i in tqdm(range(length)):
        if answer_data[i]["flag"] == 1:
            # 有效回答
            correct += 1
    true_acc = correct / length
    return true_acc

def gptscore(qus, golden, ans):
    """调用 OpenAI 评分 API"""
    prompt = f"""
    你是一位精通古诗词的专家，擅长评估回答的准确性和完整性。请根据输入的古诗词诗句内容、标注答案和生成答案进行评分。  
    任务描述：
    - 针对的是古诗词诗句是否包含典故以及典故的解释，请根据标注答案和生成答案进行评分，返回一个 1-5 之间的整数分数。
    ## 评分规则：
    - **5 分（优秀）**：内容全面，准确理解题目要求，回答完整无遗漏。分析深入，观点清晰，有理有据。表达清晰，逻辑严密，语言流畅，无明显歧义或错漏。答案高度匹配标准答案，几乎无缺漏或偏差，核心观点和分析方式基本一致。  
    - **4 分（良好）**：内容基本完整，涵盖主要信息，但可能略有遗漏。分析较深入，能进行较好的解释和分析，但论述深度稍有欠缺。表达较清晰，逻辑基本通顺，语言较为流畅。答案与标准答案较为接近，但可能遗漏部分关键点，或表达上稍有欠缺。  
    - **3 分（中等）**：内容有一定准确性，涉及题目核心内容，但关键点不够充分。分析一般，论据较为薄弱，表达基本清楚，但逻辑性一般。答案部分匹配标准答案，但缺少较多重要信息，或有轻微理解偏差，分析较浅显。  
    - **2 分（较差）**：内容较为片面，可能有明显偏差。分析缺乏，基本没有逻辑推理，仅做简单描述或复述。表达较混乱，语句不连贯。答案与标准答案存在较大偏差，可能只有少量正确点，或回答表面化，缺乏有效分析。  
    - **1 分（极差）**：内容错误或无关，回答与题目无关或基本错误。无分析，仅简单堆砌词语或含糊回答。表达困难，语言不通顺。答案与标准答案完全不匹配，可能是无关内容、完全错误的理解，或根本没有回答问题。  

    ## 评分要求：
    - 依据标注答案评估生成答案的**准确性、完整性和表达质量**。  
    - **仅返回 JSON 格式的评分结果，不要额外输出解释或分析。**  

    ## 输入：
    - **古诗词诗句内容**：{qus}
    - **标注答案**：{golden}  
    - **生成答案**：{ans}  

    ## 返回格式：
    {{
        "score": X
    }}
    （X 为 1-5 之间的整数）
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
            result = json.loads(json_response)
            return result["score"]  # 直接返回评分值，而不是字典
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}，返回数据: {raw_response}")
        except Exception as e:
            print(f"处理数据时发生错误: {e}")
    return 0  # 若三次都失败，则返回 0


def calc_gpt_score(golden_data, ans_data):
    """计算 GPT 分数"""
    count = 0  # 大于等于 2 分的个数
    length = len(golden_data)
    gpt_score = []
    total_score = 0  # 总分
    for i in tqdm(range(length)):
        if ans_data[i]["flag"] == 1:
            # 有效回答
            content = golden_data[i]["str"]
            golden = golden_data[i]["exp"]
            ans = ans_data[i]["exp"]
            score = gptscore(content, golden, ans)
            total_score += score
            gpt_score.append(score)
            if score > 2:
                count += 1
    print(count)
    true_acc = count / len(golden_data) # 计算准确率
    avg_score = total_score / len(golden_data)   # 计算平均分
    return gpt_score, true_acc, avg_score

def main():
    ans_path = r"C:\Users\HIT\Desktop\data_fina\diangu\vllm-qwen2.5-7b_lora-diangu3.json"
    output_path = ans_path.replace(".json", "_eval.json")
    golden_data = readjson(golden_path)
    ans_data = readjson(ans_path)
    
    # 计算准确率
    flase_acc = acc(golden_data=golden_data, answer_data=ans_data)
    
    # 计算 GPT 分数
    gpt_score, true_acc, avg_score = calc_gpt_score(golden_data=golden_data, ans_data=ans_data)
    print(true_acc)
    # 结果保存到 output_path
    result = {
        "accuracy": true_acc,  # 有效回答的准确率
        "average_score": avg_score,  # GPT 评分的平均分
        "scores": gpt_score,  # 每个回答的得分
        "false_accuracy": flase_acc,  # 错误回答的准确率
    }
    
    # 保存结果到文件
    save_json(result, output_path)
    
    print(f"准确率：{true_acc}")
    print(f"平均分：{avg_score}") 
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
    print("评分完成！")