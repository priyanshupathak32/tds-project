import os
import aiohttp
import base64
import sqlite3
import asyncio
import logging
import traceback
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key and DB path
API_KEY = os.environ.get("API_KEY")
DB_PATH = os.environ.get("DB_PATH", "knowledge_base.db")

# Initialize app
app = FastAPI()

# Allow all CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic input model
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

# Database connection
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Embedding generation
async def get_embedding(text):
    try:
        url = "https://aipipe.org/openai/v1/embeddings"
        headers = {
            "Authorization": API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "model": "text-embedding-3-small",
            "input": text
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["data"][0]["embedding"]
                else:
                    raise Exception(f"Embedding API error: {await response.text()}")
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise

# Content similarity search
async def find_similar_content(query_embedding, conn, top_k=5):
    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT 'discourse' as source, post_id, chunk_index, url, content, embedding 
        FROM discourse_chunks WHERE embedding IS NOT NULL
        UNION ALL
        SELECT 'markdown' as source, doc_title, chunk_index, url, content, embedding 
        FROM markdown_chunks WHERE embedding IS NOT NULL
        """)
        rows = cursor.fetchall()

        # Convert string embedding to list
        results = []
        for row in rows:
            db_embedding = list(map(float, row["embedding"].split(",")))
            similarity = sum(a * b for a, b in zip(query_embedding, db_embedding))
            results.append({
                "source": row["source"],
                "post_id": row["post_id"] if row["source"] == "discourse" else None,
                "title": row["doc_title"] if row["source"] == "markdown" else None,
                "chunk_index": row["chunk_index"],
                "url": row["url"],
                "content": row["content"],
                "score": similarity
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    except Exception as e:
        logger.error(f"Error finding similar content: {e}")
        raise

# Enrich with adjacent chunks
async def enrich_with_adjacent_chunks(conn, results):
    try:
        cursor = conn.cursor()
        enriched_results = []

        for result in results:
            enriched = result.copy()
            additional = ""

            if result["source"] == "discourse":
                post_id = result["post_id"]
                idx = result["chunk_index"]
                if idx > 0:
                    cursor.execute("SELECT content FROM discourse_chunks WHERE post_id = ? AND chunk_index = ?", (post_id, idx - 1))
                    row = cursor.fetchone()
                    if row: additional += row["content"] + " "
                cursor.execute("SELECT content FROM discourse_chunks WHERE post_id = ? AND chunk_index = ?", (post_id, idx + 1))
                row = cursor.fetchone()
                if row: additional += " " + row["content"]
            else:
                title = result["title"]
                idx = result["chunk_index"]
                if idx > 0:
                    cursor.execute("SELECT content FROM markdown_chunks WHERE doc_title = ? AND chunk_index = ?", (title, idx - 1))
                    row = cursor.fetchone()
                    if row: additional += row["content"] + " "
                cursor.execute("SELECT content FROM markdown_chunks WHERE doc_title = ? AND chunk_index = ?", (title, idx + 1))
                row = cursor.fetchone()
                if row: additional += " " + row["content"]

            if additional:
                enriched["content"] = f"{result['content']} {additional}"
            enriched_results.append(enriched)

        return enriched_results
    except Exception as e:
        logger.error(f"Error enriching chunks: {e}")
        raise

# Answer generation
async def generate_answer(question, results, max_retries=2):
    retries = 0
    while retries < max_retries:
        try:
            context = ""
            for r in results:
                context += f"\n\n{'Discourse post' if r['source'] == 'discourse' else 'Documentation'} (URL: {r['url']}):\n{r['content'][:1500]}"

            prompt = f"""Answer the following question based ONLY on the provided context. If unsure, say "I don't have enough information."
            
Context:
{context}

Question: {question}

Return in this format:
1. Answer
2. Sources:
   1. URL: [exact_url], Text: [quote or short summary]
"""

            headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that always includes exact sources."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }

            async with aiohttp.ClientSession() as session:
                async with session.post("https://aipipe.org/openai/v1/chat/completions", headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        await asyncio.sleep(2 * (retries + 1))
                        retries += 1
                    else:
                        raise Exception(await response.text())
        except Exception as e:
            logger.error(f"LLM error: {e}")
            retries += 1
            await asyncio.sleep(2)
    raise HTTPException(status_code=500, detail="Failed to generate answer")

# Parse response
def parse_llm_response(response):
    try:
        parts = response.split("Sources:", 1)
        if len(parts) == 1:
            for h in ["Reference:", "References:", "Source:"]:
                if h in response:
                    parts = response.split(h, 1)
                    break

        answer = parts[0].strip()
        links = []
        if len(parts) > 1:
            lines = parts[1].strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r'^\d+\.\s*', '', line)
                url_match = re.search(r'(http[^\s\],)]+)', line)
                text_match = re.search(r'Text:\s*["\[]?(.*?)["\]]?$', line)
                url = url_match.group(1) if url_match else None
                text = text_match.group(1) if text_match else "Source"
                if url:
                    links.append({"url": url, "text": text})
        return {"answer": answer, "links": links}
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        return {"answer": response, "links": []}

# Image + question → embedding
async def process_multimodal_query(question, image_base64):
    if not image_base64:
        return await get_embedding(question)

    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    image_content = f"data:image/jpeg;base64,{image_base64}"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": f"What do you see here? How is it relevant to: {question}"},
                {"type": "image_url", "image_url": {"url": image_content}}
            ]}
        ]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post("https://aipipe.org/openai/v1/chat/completions", headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                description = result["choices"][0]["message"]["content"]
                combined = f"{question}\nImage context: {description}"
                return await get_embedding(combined)
            else:
                return await get_embedding(question)

# POST /query
@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    try:
        if not API_KEY:
            raise HTTPException(status_code=500, detail="API_KEY not set")

        conn = get_db_connection()
        embedding = await process_multimodal_query(request.question, request.image)
        similar = await find_similar_content(embedding, conn)
        if not similar:
            return {"answer": "I couldn't find any relevant information.", "links": []}
        enriched = await enrich_with_adjacent_chunks(conn, similar)
        response = await generate_answer(request.question, enriched)
        result = parse_llm_response(response)

        if not result["links"]:
            urls = set()
            result["links"] = []
            for r in similar[:5]:
                if r["url"] not in urls:
                    urls.add(r["url"])
                    result["links"].append({
                        "url": r["url"],
                        "text": r["content"][:100] + "..." if len(r["content"]) > 100 else r["content"]
                    })
        return result
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# GET /health
@app.get("/health")
async def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        d_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        m_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        d_embed = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        m_embed = cursor.fetchone()[0]
        conn.close()

        return {
            "status": "healthy",
            "api_key_set": bool(API_KEY),
            "discourse_chunks": d_count,
            "markdown_chunks": m_count,
            "discourse_embeddings": d_embed,
            "markdown_embeddings": m_embed
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})

# Only for local run (ignored on Vercel)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
