from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/")
async def handle_request(request: Request):
    data = await request.json()

    # OpenAI-style message format
    messages = data.get("messages", [])
    question = messages[-1].get("content", "No question provided.") if messages else "No question provided."

    # Dummy response logic
    link = "https://dummyimage.com/"
    answer = f"This is an answer to: {question}\n\nLinks: {link}"

    return JSONResponse({
        "answer": answer,
        "links": [link],
        "choices": [
            {
                "message": {
                    "content": answer
                }
            }
        ]
    })
