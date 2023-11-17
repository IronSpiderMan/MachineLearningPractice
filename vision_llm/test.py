import streamlit as st

# pip install streamlit

history = st.session_state.get('history')
if not history:
    history = []

st.title("图文问答")
st.sidebar.write("你好")
st.sidebar.button("clear")
for role, msg in history:
    st.chat_message(role).write(msg)

if text := st.chat_input("请输入内容"):
    # assistant
    st.chat_message('user').write(text)
    history.append(['user', text])
    st.chat_message('assistant').write("回复：" + text)
    history.append(['assistant', "回复：" + text])
    st.session_state['history'] = history

"xxxx"
"根据上面的描述，决定下一步动作："
"1. 前进"
"2. 后退"
"3. 左转"
"...."
