import os
import sqlite3
import re
import traceback
import aiohttp
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
API_KEY = os.getenv("API_KEY")
DB_PATH = os.getenv("DB_PATH", "knowledge_base.db")

# FastAPI instance
app = FastAPI()

# Pydantic schema
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64-encoded image string

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Dummy embedding function (replace with actual implementation)
async def get_embedding(text):
    return [0.1] * 1536

# Dummy similarity function (replace with actual implementation)
async def find_similar_content(query_embedding, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM discourse_chunks LIMIT 5")
    return [dict(row) for row in cursor.fetchall()]

# Function to enrich results with adjacent chunks
async def enrich_with_adjacent_chunks(conn, results):
    try:
        logger.info(f"Enriching {len(results)} results with adjacent chunks")
        cursor = conn.cursor()
        enriched_results = []

        for result in results:
            enriched_result = result.copy()
            additional_content = ""

            if result["source"] == "discourse":
                post_id = result["post_id"]
                current_chunk_index = result["chunk_index"]
                if current_chunk_index > 0:
                    cursor.execute("""
                        SELECT content FROM discourse_chunks 
                        WHERE post_id = ? AND chunk_index = ?
                    """, (post_id, current_chunk_index - 1))
                    prev_chunk = cursor.fetchone()
                    if prev_chunk:
                        additional_content += prev_chunk["content"] + " "
                cursor.execute("""
                    SELECT content FROM discourse_chunks 
                    WHERE post_id = ? AND chunk_index = ?
                """, (post_id, current_chunk_index + 1))
                next_chunk = cursor.fetchone()
                if next_chunk:
                    additional_content += next_chunk["content"]

            elif result["source"] == "markdown":
                title = result["title"]
                current_chunk_index = result["chunk_index"]
                if current_chunk_index > 0:
                    cursor.execute("""
                        SELECT content FROM markdown_chunks 
                        WHERE doc_title = ? AND chunk_index = ?
                    """, (title, current_chunk_index - 1))
                    prev_chunk = cursor.fetchone()
                    if prev_chunk:
                        additional_content += prev_chunk["content"] + " "
                cursor.execute("""
                    SELECT content FROM markdown_chunks 
                    WHERE doc_title = ? AND chunk_index = ?
                """, (title, current_chunk_index + 1))
                next_chunk = cursor.fetchone()
                if next_chunk:
                    additional_content += next_chunk["content"]

            if additional_content:
                enriched_result["content"] = f"{result['content']} {additional_content}"
            enriched_results.append(enriched_result)
        return enriched_results
    except Exception as e:
        logger.error(f"Error in enrich_with_adjacent_chunks: {e}")
        logger.error(traceback.format_exc())
        raise

# Function to generate answer using LLM
async def generate_answer(question, relevant_results, max_retries=2):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")
    
    retries = 0
    while retries < max_retries:
        try:
            context = ""
            for result in relevant_results:
                source_type = "Discourse post" if result["source"] == "discourse" else "Documentation"
                context += f"\n\n{source_type} (URL: {result['url']}):\n{result['content'][:1500]}"
            prompt = f"""Answer the following question based ONLY on the provided context...
            
Context:
{context}

Question: {question}

Return your response in this exact format:
1. A comprehensive yet concise answer
2. Sources:
1. URL: [exact_url], Text: [quote or summary]
"""
            url = "https://aipipe.org/openai/v1/chat/completions"
            headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant..."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        await asyncio.sleep(2 * (retries + 1))
                        retries += 1
                    else:
                        raise HTTPException(status_code=response.status, detail=await response.text())
        except Exception as e:
            logger.error(f"generate_answer failed: {e}")
            retries += 1
            await asyncio.sleep(2)
    raise HTTPException(status_code=500, detail="Failed after retries")

# Parse LLM response
def parse_llm_response(response):
    try:
        parts = response.split("Sources:", 1)
        if len(parts) == 1:
            for heading in ["Source:", "References:", "Reference:"]:
                if heading in response:
                    parts = response.split(heading, 1)
                    break
        answer = parts[0].strip()
        links = []
        if len(parts) > 1:
            for line in parts[1].strip().split("\n"):
                line = re.sub(r'^\d+\.\s*|- ', '', line.strip())
                url_match = re.search(r'(https?://[^\s,]+)', line)
                text_match = re.search(r'Text:\s*(.*)', line)
                if url_match:
                    url = url_match.group(1)
                    text = text_match.group(1).strip() if text_match else "Source reference"
                    links.append({"url": url, "text": text})
        return {"answer": answer, "links": links}
    except Exception:
        return {"answer": "Error parsing response.", "links": []}

# Handle multimodal queries
async def process_multimodal_query(question, image_base64):
    if not image_base64:
        return await get_embedding(question)
    
    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    image_content = f"data:image/jpeg;base64,{image_base64}"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Look at this image and tell me what you see related to this question: {question}"},
                {"type": "image_url", "image_url": {"url": image_content}}
            ]
        }]
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    description = result["choices"][0]["message"]["content"]
                    return await get_embedding(f"{question}\nImage context: {description}")
    except Exception as e:
        logger.warning(f"Failed to process image: {e}")
    return await get_embedding(question)

# Main query endpoint
@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    if not API_KEY:
        return JSONResponse(status_code=500, content={"error": "API_KEY not set"})
    
    conn = get_db_connection()
    try:
        query_embedding = await process_multimodal_query(request.question, request.image)
        relevant_results = await find_similar_content(query_embedding, conn)
        if not relevant_results:
            return {"answer": "I couldn't find any relevant information.", "links": []}
        enriched_results = await enrich_with_adjacent_chunks(conn, relevant_results)
        llm_response = await generate_answer(request.question, enriched_results)
        result = parse_llm_response(llm_response)
        if not result["links"]:
            unique_urls = set()
            for res in relevant_results[:5]:
                url = res["url"]
                if url not in unique_urls:
                    unique_urls.add(url)
                    snippet = res["content"][:100] + "..." if len(res["content"]) > 100 else res["content"]
                    result["links"].append({"url": url, "text": snippet})
        return result
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        conn.close()

# Health check
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
            "api_key_set": bool(API_KEY),
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "discourse_embeddings": discourse_embeddings,
            "markdown_embeddings": markdown_embeddings
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})

# Required for Vercel: expose as ASGI handler
handler = app
