# 一、配置ChatGLM

## 1.1 下载项目

```bash
git clone https://github.com/THUDM/ChatGLM-6B.git
```

## 1.2 下载依赖

进入项目：

```bash
cd ChatGLM-6B
```

下载依赖：

```bash
pip install -r requiremnets.txt
```

## 1.3 下载模型（16G内存）

运行如下代码即可自动下载：

```bash
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
```

或者手动下载：
https://huggingface.co/THUDM/chatglm-6b

## 1.4 下载模型（低内存）

运行如下代码即可自动下载：

```bash
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).quantize(4).half().cuda()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
```

或者手动下载：
https://huggingface.co/THUDM/chatglm-6b-int4

手动下载需要修改"THUDM/chatglm-6b-int4"为你的模型路径。

## 1.5 运行api（12G显存）

在ChatGLM-6B项目根目录，运行下面cmd

```bash
python api.py
```

## 1.6 运行api（6G显存）

修改：

```python
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, resume_download=True).half().cuda()
# 4bit量化
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, resume_download=True).quantize(
#    4).half().cuda()
# 8bit量化
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, resume_download=True).quantize(
#    8).half().cuda()
```

然后运行：

```bash
python api.py
```

# 二、运行ChatGLM_Document_Qa

## 2.1 安装环境

```bash
pip install -r requirements_glmqa.txt
```

## 2.2 准备文件

修改chatglm_document_qa.py第67行代码：

```python
documents = load_documents("你的文档目录")
```

## 2.3 运行项目

```bash
python chatglm_document_qa.py
```


