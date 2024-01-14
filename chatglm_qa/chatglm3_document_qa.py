"""
相关资料：
    llama-cpp-python文档：https://llama-cpp-python.readthedocs.io/en/latest/

前提：
    1.安装C++环境
        https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/
        勾选“使用C++桌面开发”
    2.安装模块
        pip install llama-cpp-python
        pip install llama-cpp-python[server]
    3.运行服务
        python3 -m llama_cpp.server --model “模型路径”
        # http://localhost:8000/v1
"""
import time

import os
import gradio as gr
from langchain.document_loaders import DirectoryLoader
from langchain.llms import ChatGLM
from langchain.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 加载embedding
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "shibing624/text2vec-base-chinese",
}


def load_documents(directory="documents"):
    """
    加载books下的文件，进行拆分
    :param directory:
    :return:
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    split_docs = text_spliter.split_documents(documents)
    return split_docs


def load_embedding_model(model_name="ernie-tiny"):
    """
    加载embedding模型
    :param model_name:
    :return:
    """
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceEmbeddings(
        model_name=embedding_model_dict[model_name],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    """
    讲文档向量化，存入向量数据库
    :param docs:
    :param embeddings:
    :param persist_directory:
    :return:
    """
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db


# 加载embedding模型
embeddings = load_embedding_model('text2vec3')
# 加载数据库
if not os.path.exists('VectorStore'):
    documents = load_documents()
    db = store_chroma(documents, embeddings)
else:
    db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)
# 创建llm
# llm = ChatGLM(
#     endpoint_url='http://127.0.0.1:8000',
#     max_token=80000,
#     top_p=0.9
# )
llm = LlamaCpp(
    model_path=r"G:\models\llama2\llama-2-7b-chat-q4\llama-2-7b-chat.Q4_0.gguf",
    n_ctx=2048,
    stop=['Human:']
)
# 创建qa
QA_CHAIN_PROMPT = PromptTemplate.from_template("""Human:
根据下面的上下文（context）内容回答问题。
如果你不知道答案，就回答不知道，不要试图编造答案。
答案最多3句话，保持答案简介。
总是在答案结束时说”谢谢你的提问！“
{context}
问题：{question}
Assistant:
""")
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    verbose=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(history, file):
    """
    上传文件后的回调函数，将上传的文件向量化存入数据库
    :param history:
    :param file:
    :return:
    """
    global qa
    directory = os.path.dirname(file.name)
    documents = load_documents(directory)
    db = store_chroma(documents, embeddings)
    retriever = db.as_retriever()
    qa.retriever = retriever
    history = history + [((file.name,), None)]
    return history


def bot(history):
    """
    聊天调用的函数
    :param history:
    :return:
    """
    message = history[-1][0]
    if isinstance(message, tuple):
        response = "文件上传成功！！"
    else:
        response = qa({"query": message})['result']
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=['txt'])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.queue()
if __name__ == "__main__":
    demo.launch()
