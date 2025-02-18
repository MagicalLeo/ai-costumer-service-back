from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama  # 請先確保已安裝 ollama 的 Python 客戶端

app = FastAPI()

# 定義請求資料結構
class MessageRequest(BaseModel):
    context: str
    content: str

@app.post("/backend/api/send")
async def send_message(req: MessageRequest):
    prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
    Write a response that appropriately completes the request.
    Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
    
    ### Instruction:
    You are a smart virtual assistant - Aivara, with advanced skills in problem-solving, guidance, and personalized support.
    If the customer's inquiry includes context, please provide a more detailed and comprehensive explanation, addressing all aspects of the provided context.
    Please deliver clear, insightful, and step-by-step responses that cover the underlying details.
    
    ### Context:
    {}
    
    ### Question:
    {}
    
    ### Response:
    <think>{}"""

    formatted_prompt = prompt_style.format(req.context, req.content, "")
    print(formatted_prompt)
    try:
        # 呼叫 Ollama 的 API，使用 minicpm-v 模型
        response = ollama.chat(
            model="aivara-model-14B",
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ]
        )
        # 從返回結果中取出模型回應內容
        model_response = response.get("message", {}).get("content", "")
        return {"response": model_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backend/api/hello")
async def send_message():
    model_response = "Hello world!"
    return {"response": model_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
