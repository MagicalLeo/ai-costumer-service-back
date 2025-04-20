from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import ollama
import asyncio
import json
import re

app = FastAPI()

# 定義請求資料結構
class MessageRequest(BaseModel):
    context: str
    content: str

# 原有的串流聊天API端點
@app.post("/backend/api/send")
async def send_message(req: MessageRequest):
    prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
    Write a response that appropriately completes the request.
    Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
    ### Instruction:
    You are a smart virtual assistant - Aivara, Desgined by Codebat, with advanced skills in problem-solving, guidance, and personalized support.
    If the customer's inquiry includes context, please provide a more detailed and comprehensive explanation, addressing all aspects of the provided context.
    Please deliver clear, insightful, and step-by-step responses that cover the underlying details.
    ### Context:
    {}
    ### Question:
    {}
    ### Response:"""
    
    formatted_prompt = prompt_style.format(req.context, req.content)
    print("received prompt: ", req.content)
    
    async def generate():
        try:
            # 使用 ollama 的流式輸出
            for chunk in ollama.chat(
                model="aivara-model-14B",
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                stream=True
            ):
                # 從每個片段中取出內容
                chunk_content = chunk.get('message', {}).get('content', '')
                if chunk_content:
                    # 將每個 chunk 轉為 SSE 格式
                    yield f"data: {json.dumps({'chunk': chunk_content})}\n\n"
                    # 小延遲以確保流式傳輸正常
                    await asyncio.sleep(0.01)
            
            # 發送完成信號
            yield f"data: {json.dumps({'done': True})}\n\n"
                
        except Exception as e:
            error_msg = str(e)
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
    
    # 返回 StreamingResponse，設置為 text/event-stream MIME 類型
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/backend/api/updatetitle")
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
async def hello_world():
    return {"response": "Hello world!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
