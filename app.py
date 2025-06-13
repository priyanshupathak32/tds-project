from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/")
async def handle_request(request: Request):
    data = await request.json()

    # Extract question from OpenAI-like message format
    question = data.get("messages", [{}])[-1].get("content", "No question provided.")

    # Your custom logic
    answer = f"This is an answer to: {question}\n\nLinks: https://dummyimage.com/"

    # Important: include both .answer and OpenAI-style .choices
    return JSONResponse({
        "answer": answer,
        "choices": [
            {
                "message": {
                    "content": answer
                }
            }
        ]
    })
