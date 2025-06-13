from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sqlite3
import traceback
import re
import aiohttp
import asyncio
import logging
from pydantic import BaseModel
import os

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment
API_KEY = os.getenv("API_KEY")
DB_PATH = "your_db_path_here.db"  # <-- Make sure this is correctly set

# FastAPI app
app = FastAPI()

# Enable CORS (you can restrict origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Welcome route for sanity check
@app.get("/")
async def root():
    return {"message": "Welcome to the TDS Project API. Use POST /query."}

# Request model
class QueryRequest(BaseModel):
    question: str
    image: str = None

# ... your helper functions go here ...
# enrich_with_adjacent_chunks
# generate_answer
# process_multimodal_query
# parse_llm_response
# get_db_connection
# find_similar_content

# Query endpoint
@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    try:
        logger.info(f"Received query request: question='{request.question[:50]}...', image_provided={request.image is not None}")
        
        if not API_KEY:
            error_msg = "API_KEY environment variable not set"
            logger.error(error_msg)
            return JSONResponse(status_code=500, content={"error": error_msg})
        
        conn = get_db_connection()
        try:
            query_embedding = await process_multimodal_query(request.question, request.image)
            relevant_results = await find_similar_content(query_embedding, conn)

            if not relevant_results:
                logger.info("No relevant results found")
                return {
                    "answer": "I couldn't find any relevant information in my knowledge base.",
                    "links": []
                }

            enriched_results = await enrich_with_adjacent_chunks(conn, relevant_results)
            llm_response = await generate_answer(request.question, enriched_results)
            result = parse_llm_response(llm_response)

            # If links are missing, fallback
            if not result["links"]:
                logger.info("No links extracted, creating from relevant results")
                links = []
                unique_urls = set()
                for res in relevant_results[:5]:
                    url = res["url"]
                    if url not in unique_urls:
                        unique_urls.add(url)
                        snippet = res["content"][:100] + "..." if len(res["content"]) > 100 else res["content"]
                        links.append({"url": url, "text": snippet})
                result["links"] = links

            logger.info(f"Returning result: answer_length={len(result['answer'])}, num_links={len(result['links'])}")
            return result
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"error": error_msg})
        finally:
            conn.close()
    except Exception as e:
        error_msg = f"Unhandled exception in query_knowledge_base: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": error_msg})

# Health check route
@app.get("/health")
async def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_embeddings = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_embeddings = cursor.fetchone()[0]
        conn.close()
        return {
            "status": "healthy", 
            "database": "connected", 
            "api_key_set": bool(API_KEY),
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "discourse_embeddings": discourse_embeddings,
            "markdown_embeddings": markdown_embeddings
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e), "api_key_set": bool(API_KEY)}
        )

# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
