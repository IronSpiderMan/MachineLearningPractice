import torch
import aiohttp
import asyncio
from io import BytesIO
from PIL import Image
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration


async def chat(prompt):
    async with aiohttp.ClientSession() as session:
        async with session.post('http://127.0.0.1:8000/chat', json={'prompt': prompt}) as response:
            response = await response.json()
            return response['choices'][0]['text']


prompt = ""
# 加载历史消息
messages = st.session_state.get('history_chat')
if not messages:
    messages = []
# 加载blip
model_path = r"G:\huggingface\hub\models--Salesforce--blip-image-captioning-large"
processor = st.session_state.get('processor')
if not processor:
    processor = BlipProcessor.from_pretrained(model_path)
blip = st.session_state.get('blip')
if not blip:
    blip = BlipForConditionalGeneration.from_pretrained(model_path,
                                                        torch_dtype=torch.float16).to("cuda")
# 界面
st.title("图文对话")
if file := st.file_uploader(label="请上传图片"):

    image = Image.open(BytesIO(file.getvalue()))
    st.sidebar.image(image)

    text = "a photography of"
    inputs = processor(image, text, return_tensors="pt").to("cuda", torch.float16)
    out = blip.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    st.sidebar.write(description)
    prompt += (
        "System: You need to answer the questions based on the description of the picture given below."
        "If the description has nothing to do with the question, "
        "you should just answer using your own language abilities."
        "Do not imagine non-existent facts.\n\n"
        f"Description: {description}."
    )
for role, text in messages:
    st.chat_message(role).write(text)
if message := st.chat_input("请输入问题："):
    messages.append(['user', message])
    prompt += (
        f"\n\nHuman: {message}. \n\nAssistant: "
    )
    st.chat_message('user').write(message)
    response = asyncio.run(chat(prompt))
    messages.append(['assistant', response])
    st.chat_message('assistant').write(response)
    st.session_state['history_chat'] = messages
