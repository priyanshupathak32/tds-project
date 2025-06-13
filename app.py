import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI()

# CORS setup for broader compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded string, optional

@app.post("/")
async def handle_query(req: QueryRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    # Dummy logic for response
    return {
        "answer": f"You asked: '{req.question}'",
        "image_received": bool(req.image),
        "status": "Success",
        "links": []  # ✅ Required field
    }
