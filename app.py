from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS (optional but helpful for browser testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy answer generation function (replace this with your real logic)
def generate_answer(question: str) -> str:
    # You can call your own LLM logic here
    return f"This is an answer to: {question}\n\nLinks: https://dummyimage.com/"

# Main route that mimics OpenAI's API
@app.post("/")
async def openai_compatible_route(request: Request):
    try:
        data = await request.json()
        messages = data.get("messages", [])
        user_message = messages[-1]["content"] if messages else "No question provided."

        # Call your answer-generation logic
        answer = generate_answer(user_message)

        # Return response in OpenAI format
        return JSONResponse({
            "choices": [
                {
                    "message": {
                        "content": answer
                    }
                }
            ]
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
