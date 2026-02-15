import requests
import json


class OllamaAPI:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def generate(self, model, prompt, stream=False):
        """生成文本"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

        if stream:
            response = requests.post(url, json=payload, stream=True)
            generated_text = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        generated_text += data['response']
                        print(data['response'], end='', flush=True)
                    if data.get('done'):
                        print()
                        return generated_text
        else:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                return f"Error: {response.status_code} - {response.text}"

    def chat(self, model, messages, stream=False):
        """聊天对话"""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        if stream:
            response = requests.post(url, json=payload, stream=True)
            assistant_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'message' in data and data['message']['role'] == 'assistant':
                        content = data['message']['content']
                        assistant_response += content
                        print(content, end='', flush=True)
                    if data.get('done'):
                        print()
                        return assistant_response
        else:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get('message', {}).get('content', '')
            else:
                return f"Error: {response.status_code} - {response.text}"

    def list_models(self):
        """列出可用模型"""
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        else:
            return f"Error: {response.status_code} - {response.text}"

    def pull_model(self, model):
        """拉取模型"""
        url = f"{self.base_url}/api/pull"
        payload = {
            "name": model,
            "stream": True
        }

        response = requests.post(url, json=payload, stream=True)
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                if 'status' in data:
                    print(f"Status: {data['status']}")
                    if 'digest' in data:
                        print(f"Digest: {data['digest']}")
                    if 'total' in data and 'completed' in data:
                        progress = (data['completed'] / data['total']) * 100
                        print(f"Progress: {progress:.2f}%")
            if data.get('done'):
                break
        return "Model pulled successfully!"


# 使用示例
if __name__ == "__main__":
    ollama = OllamaAPI()

    print("=== Ollama API Python Client ===")
    print()

    # 列出可用模型
    print("1. 可用模型:")
    models = ollama.list_models()
    if isinstance(models, list):
        for model in models:
            print(f"   - {model}")
    else:
        print(f"   Error: {models}")
    print()

    # 选择模型（默认使用第一个可用模型或llama3）
    if isinstance(models, list) and models:
        selected_model = models[0]
    else:
        selected_model = "llama3"
    print(f"使用模型: {selected_model}")
    print()

    # 示例1: 简单文本生成
    print("2. 示例: 简单文本生成")
    prompt = "你是谁"
    print(f"提示词: {prompt}")
    print("生成结果:")
    result = ollama.generate(selected_model, prompt)
    print(result)
    print()

    '''# 示例2: 流式文本生成
    print("3. 示例: 流式文本生成")
    prompt = "请解释一下机器学习的基本原理"
    print(f"提示词: {prompt}")
    print("生成结果 (流式):")
    result = ollama.generate(selected_model, prompt, stream=True)
    print()

    # 示例3: 聊天对话
    print("4. 示例: 聊天对话")
    messages = [
        {"role": "user", "content": "你好，我是一名学生"},
        {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
        {"role": "user", "content": "请推荐一本学习Python的好书"}
    ]
    print("聊天历史:")
    for msg in messages:
        print(f"   {msg['role']}: {msg['content']}")
    print("助理回复:")
    result = ollama.chat(selected_model, messages)
    print(result)
    print()

    # 示例4: 流式聊天对话
    print("5. 示例: 流式聊天对话")
    messages = [
        {"role": "user", "content": "你能告诉我如何学习编程吗？"}
    ]
    print(f"用户: {messages[0]['content']}")
    print("助理回复 (流式):")
    result = ollama.chat(selected_model, messages, stream=True)
    print()'''

    print("=== 演示结束 ===")