
# 宋词问答数据集处理脚本
import os
import json
from openai import OpenAI
from tqdm import tqdm
import re
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
你是一位精通古诗词的专家，现在需要你根据提供的古诗内容，准确填补缺失的句子。  

### **任务描述**  
我会提供一个 JSON 数据，格式如下：  
- **"que"**：古诗词填空题，包含缺失的诗句部分（以"________"表示）  
- **"que"**：{data}

你的任务是：  
1. 理解古诗词的内容与背景，并根据上下文填充缺失的诗句。  
2. **只输出 JSON 格式的答案**，不要额外的解释或其他无关内容。
3.用中文古诗词回答  

### **JSON 输出格式示例**  

```json
{{
    "ans": [
        " ", //第一个空的答案
        "" //第二个空的答案
    ]
}}
```
### **注意事项**
1. **仅返回 JSON 结果，不需要额外解释或输出其他内容。**
2. **请确保 "ans" 只输出正确答案的列表，确保ans内容是中文不要附加其他文本。**
3. **请保持输出格式整洁，符合 JSON 规范。确保json文件ans回答是中文**
4. **不要给出你的解释、思考**
5. **请确保 "ans" 中的每个答案都与 "que" 中的缺失部分一一对应。**
6.**如果你不会做，答案为空**
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
            content = response.choices[0].message.content
            print(content)
            # print(test)
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                answer = match.group(0)  # 获取匹配的 JSON 字符串
                answer = answer.strip()  # 去除前后空格

                # 解析 JSON
                try:
                    answer = json.loads(answer)
                    return answer
                except json.JSONDecodeError as e:
                    print(f"JSON 解析失败，错误: {e}，重试中...")
                    continue

        except Exception as e:
            print(f"处理数据时发生错误: {e}，重试中...")
    return {
                    "que": data,
                    "ans": ["-1"]
                }
def main():
    # 读取输入数据
    output_path = r"/mnt/home/user04/CCL/tiankong/vllm-{}-tiankong3.json".format(model)
    with open (r'/mnt/home/user04/CCL/tiankong/data.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    output_data = []
    for data in tqdm(input_data, desc="Processing"):
        # print(data["que"])
        answer = get_response(data["que"])
        # print(answer)
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