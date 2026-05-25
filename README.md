# RAG Chat Assistant

Production-grade Retrieval-Augmented Generation chat assistant built with FastAPI, Sentence Transformers, FAISS, OpenRouter, and a vanilla HTML/CSS/JavaScript frontend.

## Architecture

```text
Frontend (HTML/CSS/JS)
        |
        v
FastAPI Backend
        |
        v
RAG Pipeline
  - Query Embedding
  - FAISS Cosine Similarity Search
  - Top-K Retrieval (K=3)
  - Similarity Threshold (0.65)
  - Prompt Builder with Context + History
  - OpenRouter Chat Completions API
        |
        v
Grounded Response
```

## RAG Workflow

1. Documents are loaded from `data/docs.json`.
2. Documents are chunked with overlap and metadata.
3. `sentence-transformers/paraphrase-MiniLM-L3-v2` creates local embeddings from each chunk plus its title, source metadata, and FAQ-style question anchors.
4. Embeddings are normalized and stored in FAISS using inner product search, which is equivalent to cosine similarity for normalized vectors.
5. `/api/chat` embeds the user question and retrieves the top 3 chunks.
6. If the best similarity score is below `0.65`, the API returns:

```text
I could not find enough information in the knowledge base.
```

7. If retrieval passes the threshold, the prompt includes retrieved context and the last 5 session messages.
8. OpenRouter sends the grounded prompt to the configured model and returns the final answer.

## Project Structure

```text
rag-chatbot/
|-- app/
|   |-- main.py
|   |-- config.py
|   |-- routes/
|   |   `-- chat.py
|   |-- services/
|   |   |-- embeddings.py
|   |   |-- retriever.py
|   |   |-- llm.py
|   |   |-- rag_pipeline.py
|   |   `-- memory.py
|   |-- vectorstore/
|   |   `-- faiss_store.py
|   |-- utils/
|   |   |-- chunking.py
|   |   `-- loaders.py
|   `-- prompts/
|       `-- prompt_template.py
|-- frontend/
|   |-- index.html
|   |-- style.css
|   `-- app.js
|-- data/
|   `-- docs.json
|-- requirements.txt
|-- .env.example
|-- README.md
`-- render.yaml
```

## API

### `GET /health`

Returns service and index status.

### `POST /api/chat`

Request:

```json
{
  "message": "How do I reset my password?",
  "session_id": "browser-session-id"
}
```

Response:

```json
{
  "answer": "You can reset your password from the sign in page...",
  "session_id": "browser-session-id",
  "sources": [
    {
      "title": "Password Reset",
      "source": "help-center/password-reset",
      "chunk_id": 0,
      "similarity": 0.8123
    }
  ],
  "used_fallback": false,
  "similarity_score": 0.8123
}
```

## Local Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```bash
copy .env.example .env
```

Add your OpenRouter API key:

```text
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=google/gemini-2.0-flash-exp:free
```

Run the backend:

```bash
uvicorn app.main:app --reload
```

Open `frontend/index.html` in a browser. For deployment, update `API_URL` in `frontend/app.js` to your Render backend URL.

## OpenRouter LLM

The backend calls OpenRouter's OpenAI-compatible chat completions endpoint at:

```text
https://openrouter.ai/api/v1/chat/completions
```

It sends:

- `Authorization: Bearer <OPENROUTER_API_KEY>`
- `HTTP-Referer`
- `X-OpenRouter-Title`

The default model is `google/gemini-2.0-flash-exp:free`. You can replace it with any OpenRouter chat model by changing `OPENROUTER_MODEL`.

## Embedding Strategy

The app uses `paraphrase-MiniLM-L3-v2`, a fast local embedding model from Sentence Transformers. It avoids paid embedding APIs and works well for small to medium knowledge bases. Each document chunk is indexed along with FAQ-style question anchor vectors that point back to the same source content. This improves short natural-language queries like "How do I get a refund?" while answers remain grounded in the original content. Embeddings are normalized before being inserted into FAISS.

## Similarity Search

FAISS uses `IndexFlatIP`. Because embeddings are normalized, inner product scores represent cosine similarity. The retriever searches a wider candidate set, removes duplicate source chunks, returns the top 3 unique chunks, and logs similarity scores for observability.

## Conversation History

Session history is stored in memory with a maximum of 5 messages per session. The frontend stores a stable `session_id` in `localStorage`, and the backend uses that ID to retrieve recent conversation turns.

## Error Handling

The backend handles:

- Missing or invalid OpenRouter API keys with `401`
- Rate limits with `429`
- OpenRouter timeouts with `504`
- Validation errors through FastAPI and Pydantic

## Deployment

### Backend on Render

1. Push this project to GitHub.
2. Create a new Render Web Service.
3. Use `render.yaml` or set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable `OPENROUTER_API_KEY`.
5. Add your frontend domain to `ALLOWED_ORIGINS`.
6. Set `OPENROUTER_SITE_URL` to your deployed backend or frontend URL.

### Frontend on Vercel

1. Deploy the `frontend` folder as a static project.
2. Set `API_URL` in `frontend/app.js` to your Render URL:

```js
const API_URL = "https://your-render-service.onrender.com";
```

## Screenshots

Add screenshots here after running the application:

- Chat interface
- Successful grounded answer
- Fallback response for out-of-scope question

## Notes

- This implementation performs real embedding retrieval, not keyword matching.
- The model is instructed to use only retrieved context.
- The FAISS index is built in memory at startup, which works well for Render free tier and the included sample knowledge base.
