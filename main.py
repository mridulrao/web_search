import argparse
import json
import os
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fetch_content import ContentFetcher
from query_expansion import DEFAULT_MODEL, QueryExpansionConfig, QueryExpander, build_client
from search_retrieval import SearchRetriever
from url_reranking import OpenAIEmbeddingClient, UrlReranker, UrlRerankingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand a search query and retrieve URLs from search providers.")
    parser.add_argument("query", help="The original user search query to expand.")
    parser.add_argument(
        "--expansion-count",
        type=int,
        default=5,
        help="Number of expanded queries to generate. Default: 5.",
    )
    parser.add_argument(
        "--results-per-provider",
        type=int,
        default=5,
        help="Number of URLs to retrieve per provider for each query. Default: 5.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of final ranked URLs to return. Default: 5.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"OpenAI model to use. Default: OPENAI_MODEL or {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embedding model to use for URL reranking.",
    )
    return parser.parse_args()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    expansion_count: int = Field(default=5, ge=1)
    results_per_provider: int = Field(default=5, ge=1)
    top_k: int = Field(default=5, ge=1)
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    embedding_model: str = Field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))


class UrlRequest(BaseModel):
    url: str = Field(min_length=1)


app = FastAPI(title="Web Search API")


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


def run_query_pipeline(
    query: str,
    expansion_count: int = 5,
    results_per_provider: int = 5,
    top_k: int = 5,
    model: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, object]:
    expander = QueryExpander(
        client=build_client(),
        config=QueryExpansionConfig(
            model=model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            expansion_count=expansion_count,
        ),
    )
    expanded_queries = expander.expand(query)
    retriever = SearchRetriever()
    search_results = retriever.search_queries(
        queries=expanded_queries,
        limit_per_provider=results_per_provider,
    )
    reranker = UrlReranker(
        embedding_client=OpenAIEmbeddingClient(
            client=build_client(),
            model=embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        ),
        config=UrlRerankingConfig(
            embedding_model=embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            top_k=top_k,
        ),
    )
    ranked_urls = reranker.rerank_search_payload(
        search_payload=search_results,
        top_k=top_k,
    )
    content_fetcher = ContentFetcher()
    fetched_pages = []
    with ThreadPoolExecutor(max_workers=min(max(len(ranked_urls), 1), 8)) as executor:
        fetch_results = list(executor.map(content_fetcher.fetch, [ranked_url.url for ranked_url in ranked_urls]))

    for ranked_url, fetch_result in zip(ranked_urls, fetch_results, strict=True):
        if not fetch_result.success or fetch_result.status_code != 200:
            continue
        fetched_pages.append(
            {
                "ranking": asdict(ranked_url),
                "content": fetch_result.to_public_dict(),
            }
        )
    return {
        "original_query": query,
        "expanded_queries": expanded_queries,
        "top_k": len(fetched_pages),
        "results": fetched_pages,
    }


def fetch_url_content(url: str) -> dict[str, str]:
    fetch_result = ContentFetcher().fetch(url)
    if not fetch_result.success or fetch_result.status_code != 200:
        raise RuntimeError(fetch_result.error or f"failed to fetch URL with status {fetch_result.status_code}")
    return {
        "url": fetch_result.url,
        "final_url": fetch_result.final_url,
        "content": fetch_result.content,
    }


@app.post("/search")
def search_endpoint(request: QueryRequest) -> dict[str, object]:
    try:
        return run_query_pipeline(
            query=request.query,
            expansion_count=request.expansion_count,
            results_per_provider=request.results_per_provider,
            top_k=request.top_k,
            model=request.model,
            embedding_model=request.embedding_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/content")
def content_endpoint(request: UrlRequest) -> dict[str, str]:
    try:
        return fetch_url_content(request.url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_query_pipeline(
                query=args.query,
                expansion_count=args.expansion_count,
                results_per_provider=args.results_per_provider,
                top_k=args.top_k,
                model=args.model,
                embedding_model=args.embedding_model,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
