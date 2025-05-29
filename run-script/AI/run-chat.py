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
    你是一位精通古诗词的专家，擅长分析诗歌中的典故。请根据以下输入文本，判断其是否包含典故，并严格按照指定的 JSON 格式返回结果。

    ## 任务描述

    1. **判断输入文本是否包含典故：**  
    - 如果包含典故，则 `flag=1`，并提供典故的详细解释。  
    - 如果不包含典故，则 `flag=0`，典故解释为空字符串 `""`。  
    2. **仅返回符合指定 JSON 格式的内容，不要额外输出解释或分析。**  
    3. **请确保输出格式整洁，符合 JSON 规范。禁止出现其他内容**
    4. **请注意，你只需要给出对应的答案，不需要给出任何注释和注意和其他内容**
    ## 输入文本：
    {data}
    请严格按照以下 JSON 结构返回结果：
    {{
    "str": "<输入文本>",
    "flag": <1 或 0>,
    "exp": "<典故解释>"
    }}
    """
    
    for _ in range(3):  # 三次重传机制
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个古诗词专家，请按照要求回答问题"},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            test = response.choices[0].message.content
            print(test)
            test = test.replace("```json", "").replace("```", "").strip()
            test = test.replace('```json', '').replace('```', '').strip()
            test = test.strip().strip("```json").strip("```")

            return json.loads(test)
        except json.JSONDecodeError:
            print(f"JSON 解析失败，重试中")
            continue
        except Exception as e:
            print(f"处理数据时发生错误: {e}")
            continue
    
    # 如果三次重传都失败，返回一个空的格式，保持一致性
    return {
        "str": "",
        "flag": 0,
        "exp": ""
    }

def main():
    # 读取输入数据
    output_path =  r"/mnt/home/user04/CCL/diangu/vllm-{}-diangu3.json".format(model)
    with open(r'/mnt/home/user04/CCL/diangu/diangu.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    output_data = []
    for data in tqdm(input_data, desc="Processing"):
        answer = get_response(json.dumps(data["str"], ensure_ascii=False))
        # print(answer)
        output_data.append(answer)

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成。共处理 {len(output_data)} 条数据，输出已保存到 {output_path}。")

if __name__ == "__main__":
    main()
