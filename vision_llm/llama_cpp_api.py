# uvicorn, llama-cpp-python, fastapi
import uvicorn
from llama_cpp import Llama
from fastapi import FastAPI, Request

app = FastAPI()
llm = Llama(
    model_path=r'G:\models\llama2\llama-2-13b-chat-q4\ggml-model-q4_0.gguf',
    n_ctx=2048
)


@app.post("/chat")
async def chat(request: Request):
    global llm
    jdata = await request.json()
    prompt = jdata['prompt']
    return llm(prompt, stop=['Human'])


@app.get("/test")
async def chat():
    global llm
    return llm("你好", stop=['Human'])


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, workers=1)
