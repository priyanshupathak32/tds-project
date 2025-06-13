# app.py
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.68
MAX_RESULTS = 10
MAX_CONTEXT_CHUNKS = 4

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (optional, can restrict origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

# ✅ Working root POST endpoint
@app.post("/")
async def handle_query(req: QueryRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    logger.info(f"Received question: {req.question}")
    
    # Dummy logic – replace with actual processing
    return {
        "answer": f"You asked: '{req.question}'",
        "image_received": bool(req.image),
        "status": "Success"
    }

# Optional: Uvicorn for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
