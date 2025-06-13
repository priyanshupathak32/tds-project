from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Optional: Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace with your actual logic
def generate_answer(question: str) -> str:
    return f"This is an answer to: {question}\n\nLinks: https://dummyimage.com/"

@app.post("/")
async def openai_compatible_route(request: Request):
    try:
        data = await request.json()
        messages = data.get("messages", [])
        user_message = messages[-1]["content"] if messages else "No question provided."

        # Generate answer
        answer = generate_answer(user_message)

        # Return both "answer" and OpenAI-style "choices"
        return JSONResponse({
            "answer": answer,  # ✅ Added top-level "answer"
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
