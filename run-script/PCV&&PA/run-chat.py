import os
import json
from openai import OpenAI
from tqdm import tqdm

# 设置 HTTP 代理
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    base_url=openai_api_base,
    api_key=openai_api_key
)

models = client.models.list()
model = models.data[0].id
print(model)

def get_response(data):
    prompt = f"""
    你是一个古诗词专家，请按照要求回答问题。
    数据是一个JSON格式的对象，包含诗歌信息和问题，请根据其内容进行回答。格式如下：
    {{
        "id": <诗歌编号>,
        "title": "<诗歌标题>",
        "author": "<作者>",
        "content": "<诗歌内容>",
        "questions": [
            {{
                "question": "<问题内容>",
                "type": "选择题" | "简答题",
                "options": {{
                    "A": "<选项A>",
                    "B": "<选项B>",
                    "C": "<选项C>",
                    "D": "<选项D>"
                }}
            }}
        ]
    }}
    数据：{data}

    回答要求：
    1. 对于选择题(type="选择题")：
       - 只需返回正确选项字母，如："A"
       - 无需解释理由

    2. 对于简答题(type="简答题")：
       - 按照要求进入作答
       - 控制在50-150字以内
       - 重点:分点作答 1. 2. 3. ...

    请遵循以下规范：
    - 严格基于原文分析，避免主观臆测
    - 保持逻辑性和连贯性
    - 确保答案有充分的文本依据
    - 注意你只需要给出对应的答案，不需要给出任何注释和注意和其他内容

    请以JSON格式输出答案：
    {{
        "id": <诗歌编号>
        "answer": [
           "<问题1答案><选择题只需返回正确选项字母，不需要返回解释>",
           "<问题2答案><解答题需按照要求回答，分点作答 1. 2. 3...>",
            ...
        ]
    }}
    - 注意你只需要给出对应的答案，不需要给出任何注释和注意和其他内容
    """
    
    for _ in range(3):  # 三次重传机制
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "你是一个古诗词专家，请按照要求回答问题"},
                          {"role": "user", "content": prompt}],
                stream=False
            )
            response_content = response.choices[0].message.content
            print(response_content)
            response_content = response_content.replace("```json", "").replace("```", "").strip()

            return json.loads(response_content)
        except json.JSONDecodeError:
            print(f"JSON解析失败，重试中...")
            continue
        except Exception as e:
            print(f"处理数据时发生错误: {e}")
            continue
    
    # 三次请求都失败时返回默认格式的响应
    return {
        "id": data.get("id", ""),
        "answer": ["-1"]
    }

def main():
    # 读取输入数据
    output_path =  r"/mnt/home/user04/CCL/moniti/vllm-{}-moniti3.json".format(model)
    with open(r'/mnt/home/user04/CCL/moniti/moniti_qus.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    output_data = []
    for data in tqdm(input_data, desc="Processing"):
        answer = get_response(data)
        if answer:
            output_data.append(answer)
        else:
            print(f"未能生成答案的数据")

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"处理完成。共处理 {len(output_data)} 条数据，输出已保存到 {output_path}。")

if __name__ == "__main__":
    main()
