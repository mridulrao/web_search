# Web Search API

This project is a FastAPI-based web search and content extraction service.

Given a user query, it:

1. Expands the query with OpenAI.
2. Searches DuckDuckGo through both HTML retrieval and Firefox browser automation.
3. Reranks the collected URLs with embeddings and a cross-encoder.
4. Fetches and extracts readable content from the top-ranked pages.

It also exposes a direct URL content extraction endpoint for fetching a single page.

## Features

- `POST /search` for full web search, reranking, and content extraction
- `POST /content` for direct page content extraction from a URL
- CLI support through `main.py`
- Docker and Docker Compose deployment
- Caddy reverse proxy with automatic HTTPS for EC2 deployments

## Requirements

- Python 3.12+
- OpenAI API key
- For local non-Docker runs:
  - Firefox
  - `geckodriver`
  - Playwright Chromium dependencies if browser fallback is needed

## Environment

Create a `.env` file:

```bash
cp .env.example .env
```

Minimum required values:

```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DOMAIN=search.example.com
LETSENCRYPT_EMAIL=you@example.com
```

Optional Cloudflare browser-rendering support can also be provided through your `.env` if you use it in `fetch_content.py`.

## Local Development

Install dependencies:

```bash
pip install -e .
```

Run the API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API Endpoints

### `POST /search`

Runs the full search pipeline and returns the same result shape as the CLI.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "best electric SUVs for winter driving",
    "expansion_count": 5,
    "results_per_provider": 5,
    "top_k": 5
  }'
```

Example response shape:

```json
{
  "original_query": "best electric SUVs for winter driving",
  "expanded_queries": [
    "best electric SUVs for winter driving range and traction"
  ],
  "top_k": 1,
  "results": [
    {
      "ranking": {
        "query": "best electric SUVs for winter driving range and traction",
        "title": "Example result",
        "url": "https://example.com",
        "snippet": "",
        "provider": "duckduckgo",
        "source_type": "documentation",
        "heuristic_score": 1.0,
        "embedding_score": 0.9,
        "cross_encoder_score": 0.8,
        "combined_score": 0.87
      },
      "content": {
        "url": "https://example.com",
        "final_url": "https://example.com",
        "title": "Example result",
        "content": "Extracted page content",
        "method": "http",
        "success": true,
        "status_code": 200
      }
    }
  ]
}
```

### `POST /content`

Fetches one URL and returns the extracted content only.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/content \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com"}'
```

Example response:

```json
{
  "url": "https://example.com",
  "final_url": "https://example.com",
  "content": "Extracted page content"
}
```

### `GET /healthz`

Returns a simple container health response:

```json
{
  "status": "ok"
}
```

## CLI Usage

You can still run the full pipeline from the command line:

```bash
python main.py "best electric SUVs for winter driving"
```

With explicit options:

```bash
python main.py \
  "best electric SUVs for winter driving" \
  --expansion-count 5 \
  --results-per-provider 5 \
  --top-k 5 \
  --model gpt-4.1-mini \
  --embedding-model text-embedding-3-small
```

## Project Files

- [main.py](/Users/mridulrao/Downloads/psuedo_desktop/web_search/main.py): FastAPI app and CLI entrypoint
- [query_expansion.py](/Users/mridulrao/Downloads/psuedo_desktop/web_search/query_expansion.py): OpenAI-driven query expansion
- [search_retrieval.py](/Users/mridulrao/Downloads/psuedo_desktop/web_search/search_retrieval.py): DuckDuckGo search retrieval clients
- [url_reranking.py](/Users/mridulrao/Downloads/psuedo_desktop/web_search/url_reranking.py): embedding and cross-encoder reranking
- [fetch_content.py](/Users/mridulrao/Downloads/psuedo_desktop/web_search/fetch_content.py): page fetching and readable content extraction
- [Dockerfile](/Users/mridulrao/Downloads/psuedo_desktop/web_search/Dockerfile): application container build
- [docker-compose.yml](/Users/mridulrao/Downloads/psuedo_desktop/web_search/docker-compose.yml): app + Caddy deployment
- [Caddyfile](/Users/mridulrao/Downloads/psuedo_desktop/web_search/Caddyfile): HTTPS reverse proxy config

## Docker Deployment

Build and start locally:

```bash
docker compose up --build -d
```

The stack contains:

- `app`: the FastAPI service on internal port `8000`
- `caddy`: reverse proxy on ports `80` and `443`

## EC2 Deployment

1. Launch an EC2 instance with Docker and Docker Compose installed.
2. Point your domain's DNS `A` record to the EC2 public IP.
3. Open inbound TCP ports `80` and `443` in the EC2 security group.
4. Copy the repository to the instance.
5. Create `.env` from `.env.example` and fill in real values.
6. Start the stack:

```bash
docker compose up --build -d
```

Once DNS is live, Caddy will automatically provision and renew the TLS certificate for `DOMAIN`.

## Notes

- This service makes outbound calls to OpenAI and search targets.
- The container image installs Firefox, `geckodriver`, and Playwright Chromium because the pipeline uses browser automation and browser-based fetch fallback.
- If `fastapi`, browser dependencies, or model dependencies are missing in local development, prefer the Docker path instead of debugging the host machine first.
