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
    你是一位古诗词专家，现在有一些古诗词需要你的帮助。  
    我会给你提供一个 JSON 数据，格式如下：  
    - **"title"**：古诗词的标题  
    - **"author"**：古诗词的作者  
    - **"content"**：古诗词的内容  
    - **"qa"**：需要提供白话文译文的诗句  
    - **"choose"**：一个包含多个选项的字典，每个选项代表该诗词可能描写的节日  

    这是我的数据：{data}  

    ### **你的任务**：  
    请你根据提供的数据，生成如下 JSON 格式的结果：  
    - **"ans_qa"**：对 "qa" 诗句提供白话文译文  
    - **"choose_id"**：从 "choose" 选项中选择最符合该诗词描述的节日标号，仅输出对应的字母
    - **"注意事项"** ：不要思考、不要输出其他无关内容、不需要给出你的想法和解释
    - json 文件中应该使用双引号
    ### **JSON 输出格式示例**：  

    ```json  
    {{  
        "ans_qa": {{  
            "羞红颦浅恨，晚风未落，片绣点重茵": "娇美的红花仿佛是美人含羞的笑脸，嫩绿的叶片点缀在她的鬓边，仿佛轻蹙黛眉，微微含恨。"  
        }},  
        "choose_id": ""  
    }} 
    ```
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
            # Get the content from the response
            test = response.choices[0].message.content

            # Sanitize the response
            test = test.replace("```json", "").replace("```", "").strip()
            test = test.replace('```json', '').replace('```', '').strip()
            test = test.strip().strip("```json").strip("```")

            # Try to parse the sanitized response
            return json.loads(test)
        
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败，错误: {e}. 重试中")
            continue
        except Exception as e:
            print(f"处理数据时发生错误: {e}")
            continue
    
    # If all attempts fail, return an empty format
    return {
        "ans_qa": "",
        "choose_id": 0,
    }

def main():
    # 读取输入数据
    output_path = r"/mnt/home/user04/CCL/wenhua/vllm-{}-wenhua.json".format(model)
    print(f"输出文件路径: {output_path}")
    with open(r'/mnt/home/user04/CCL/wenhua/qus.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    output_data = []
    for data in tqdm(input_data, desc="Processing"):
        answer = get_response(json.dumps(data, ensure_ascii=False))
        print(answer)
        output_data.append(answer)

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成。共处理 {len(output_data)} 条数据，输出已保存到 {output_path}。")

if __name__ == "__main__":
    main()
