# TianWen-benchmark

**TianWen** is a benchmark designed to evaluate large language models (LLMs) on tasks involving the understanding and reasoning of Chinese classical poetry.  


## Installation

Install the required dependency:
pip install vllm
## Project Structure
run-script/<br>
├── run-chat.py      # Launches an interactive chat interface for the model<br>
└── eval.py           # Evaluates model performance on TianWen benchmark tasks

## Usage
Step 1: Download the Model


Download the base model (e.g., Qwen2.5-14B-Instruct) and place it in your local directory. Update the paths inside the scripts accordingly.


Step 2: vllm Run run-chat.py


Step 3: vllm Run eval.py 


##License
MIT License
