import aiohttp
import asyncio
from collections import Counter
from io import BytesIO
from PIL import Image
import streamlit as st
from ultralyticsplus import YOLO


async def chat(prompt):
    async with aiohttp.ClientSession() as session:
        async with session.post('http://127.0.0.1:8000/chat', json={'prompt': prompt}) as response:
            response = await response.json()
            return response['choices'][0]['text']


def location_to_description(yolo_results):
    counter = Counter()
    for category, box in zip(yolo_results[0].boxes.cls.cpu().numpy(), yolo_results[0].boxes.xywhn.cpu().numpy()):
        x, y, w, h = box
        if x < 0.33 and y < 0.33:
            location = "upper left area"
        elif x > 0.66 and y > 0.66:
            location = "lower right area"
        elif x < 0.33 and y > 0.66:
            location = "lower left area"
        elif x > 0.66 and y < 0.33:
            location = "upper right area"
        elif 0.33 < x < 0.66 < y:
            location = "bottom center area"
        elif 0.66 > x > 0.33 > y:
            location = "top center area"
        else:
            location = "center area"
        counter.update((f"There is ?? {yolo.model.names[int(category)]} located in the {location}.",))
    description = ""
    for obj in counter.most_common():
        tmp, count = obj
        if count > 1:
            tmp = tmp.replace("is", "are")
        tmp = tmp.replace("??", str(count))
        description += tmp + "\n"
    return description


prompt = ""
# 加载历史消息
messages = st.session_state.get('history_chat')
if not messages:
    messages = []
# 加载yolo
yolo = st.session_state.get('yolo')
if not yolo:
    yolo = YOLO('ultralyticsplus/yolov8s')
    st.session_state['yolo'] = yolo

# 界面
st.title("图文对话")
if file := st.file_uploader(label="请上传图片"):
    image = Image.open(BytesIO(file.getvalue()))
    st.sidebar.image(image)
    results = yolo.predict(image)
    description = location_to_description(results)
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
