from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Iterable
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder


load_dotenv()

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

PREFERRED_SOURCES = {
    "documentation": 1.0,
    "research_paper": 0.95,
    "github": 0.85,
    "medium": 0.75,
}

SOURCE_HINTS = {
    "documentation": (
        "/docs/",
        "docs.",
        "documentation",
        "developer.",
        "readthedocs.io",
        "docs.rs",
        "pkg.go.dev",
        "developer.mozilla.org",
        "platform.openai.com/docs",
    ),
    "research_paper": (
        "arxiv.org",
        "doi.org",
        "acm.org",
        "ieeexplore.ieee.org",
        "openreview.net",
        "paperswithcode.com/paper",
        "research.google",
    ),
    "github": (
        "github.com",
        "github.io",
    ),
    "medium": (
        "medium.com",
        ".medium.com",
    ),
}


@dataclass(slots=True)
class RetrievedUrl:
    query: str
    title: str
    url: str
    snippet: str = ""
    provider: str = ""


@dataclass(slots=True)
class RankedUrl:
    query: str
    title: str
    url: str
    snippet: str
    provider: str
    source_type: str
    heuristic_score: float
    embedding_score: float
    cross_encoder_score: float
    combined_score: float


@dataclass(slots=True)
class UrlRerankingConfig:
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL
    heuristic_weight: float = 0.20
    embedding_weight: float = 0.35
    cross_encoder_weight: float = 0.45
    top_k: int = 10


class OpenAIEmbeddingClient:
    def __init__(self, client: OpenAI, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.client = client
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class UrlReranker:
    def __init__(
        self,
        embedding_client: OpenAIEmbeddingClient,
        cross_encoder: CrossEncoder | None = None,
        config: UrlRerankingConfig | None = None,
    ) -> None:
        self.config = config or UrlRerankingConfig()
        self.embedding_client = embedding_client
        self.cross_encoder = cross_encoder or CrossEncoder(self.config.cross_encoder_model)

    def rerank(self, query: str, urls: Iterable[RetrievedUrl]) -> list[RankedUrl]:
        deduplicated_urls = self._deduplicate_urls(urls)
        if not deduplicated_urls:
            return []

        query_embedding = self.embedding_client.embed_texts([query])[0]
        title_embeddings = self.embedding_client.embed_texts([item.title for item in deduplicated_urls])
        cross_encoder_inputs = [
            (query, self._cross_encoder_text(item.title, item.snippet))
            for item in deduplicated_urls
        ]
        cross_encoder_scores = self.cross_encoder.predict(cross_encoder_inputs)

        ranked_urls: list[RankedUrl] = []
        for item, title_embedding, cross_encoder_score in zip(
            deduplicated_urls,
            title_embeddings,
            cross_encoder_scores,
            strict=True,
        ):
            source_type = self._classify_source(item.url)
            heuristic_score = PREFERRED_SOURCES.get(source_type, 0.2)
            embedding_score = self._cosine_similarity(query_embedding, title_embedding)
            cross_score = self._normalize_cross_encoder_score(float(cross_encoder_score))
            combined_score = (
                self.config.heuristic_weight * heuristic_score
                + self.config.embedding_weight * embedding_score
                + self.config.cross_encoder_weight * cross_score
            )
            ranked_urls.append(
                RankedUrl(
                    query=query,
                    title=item.title,
                    url=item.url,
                    snippet=item.snippet,
                    provider=item.provider,
                    source_type=source_type,
                    heuristic_score=heuristic_score,
                    embedding_score=embedding_score,
                    cross_encoder_score=cross_score,
                    combined_score=combined_score,
                )
            )

        ranked_urls.sort(key=lambda item: item.combined_score, reverse=True)
        return ranked_urls[: self.config.top_k]

    def rerank_search_payload(
        self,
        search_payload: dict[str, dict[str, list[dict[str, str]]]],
        top_k: int | None = None,
    ) -> list[RankedUrl]:
        flattened_urls: list[RetrievedUrl] = []
        queries: list[str] = []

        for query, provider_payload in search_payload.items():
            queries.append(query)
            for results in provider_payload.values():
                for item in results:
                    flattened_urls.append(
                        RetrievedUrl(
                            query=item.get("query", query),
                            title=item["title"],
                            url=item["url"],
                            snippet=item.get("snippet", ""),
                            provider=item.get("provider", ""),
                        )
                    )

        combined_query = " ".join(dict.fromkeys(queries))
        original_top_k = self.config.top_k
        if top_k is not None:
            self.config.top_k = top_k
        try:
            return self.rerank(query=combined_query, urls=flattened_urls)
        finally:
            self.config.top_k = original_top_k

    def _deduplicate_urls(self, urls: Iterable[RetrievedUrl]) -> list[RetrievedUrl]:
        deduplicated: list[RetrievedUrl] = []
        seen: set[str] = set()

        for item in urls:
            canonical_url = self._canonicalize_url(item.url)
            if not canonical_url or canonical_url in seen:
                continue
            seen.add(canonical_url)
            deduplicated.append(
                RetrievedUrl(
                    query=item.query,
                    title=item.title.strip(),
                    url=canonical_url,
                    snippet=item.snippet.strip(),
                    provider=item.provider.strip(),
                )
            )

        return deduplicated

    @staticmethod
    def _cross_encoder_text(title: str, snippet: str) -> str:
        clean_title = " ".join(title.split())
        clean_snippet = " ".join(snippet.split())
        return clean_title if not clean_snippet else f"{clean_title} {clean_snippet}"

    @staticmethod
    def _normalize_cross_encoder_score(score: float) -> float:
        return 1.0 / (1.0 + math.exp(-score))

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        cosine = numerator / (left_norm * right_norm)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    @staticmethod
    def _canonicalize_url(url: str) -> str:
        if not url:
            return ""

        parsed = urlparse(url.strip())
        if not parsed.scheme or not parsed.netloc:
            return ""

        filtered_query = []
        for key, values in parse_qs(parsed.query, keep_blank_values=True).items():
            if key.lower().startswith("utm_"):
                continue
            if key.lower() in {"ref", "source", "campaign"}:
                continue
            for value in values:
                filtered_query.append((key, value))

        query = "&".join(f"{key}={value}" for key, value in filtered_query)
        path = parsed.path.rstrip("/") or "/"
        return parsed._replace(query=query, fragment="", path=path).geturl()

    @staticmethod
    def _classify_source(url: str) -> str:
        lowered_url = url.lower()
        for source_type, hints in SOURCE_HINTS.items():
            if any(hint in lowered_url for hint in hints):
                return source_type
        return "general"


def build_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")
    return OpenAI(api_key=api_key)


def load_urls_from_json(path: str) -> tuple[str, list[RetrievedUrl]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    query = payload["query"]
    raw_urls = payload["urls"]
    urls = [
        RetrievedUrl(
            query=item.get("query", query),
            title=item["title"],
            url=item["url"],
            snippet=item.get("snippet", ""),
            provider=item.get("provider", ""),
        )
        for item in raw_urls
    ]
    return query, urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate and rerank retrieved URLs for a query.")
    parser.add_argument("input_json", help="Path to a JSON file with 'query' and 'urls'.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum number of ranked URLs to return. Default: 10.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help=f"OpenAI embedding model to use. Default: OPENAI_EMBEDDING_MODEL or {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--cross-encoder-model",
        default=DEFAULT_CROSS_ENCODER_MODEL,
        help=f"Sentence Transformers cross-encoder model. Default: {DEFAULT_CROSS_ENCODER_MODEL}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query, urls = load_urls_from_json(args.input_json)
    reranker = UrlReranker(
        embedding_client=OpenAIEmbeddingClient(
            client=build_openai_client(),
            model=args.embedding_model,
        ),
        config=UrlRerankingConfig(
            embedding_model=args.embedding_model,
            cross_encoder_model=args.cross_encoder_model,
            top_k=args.top_k,
        ),
    )
    ranked = reranker.rerank(query=query, urls=urls)
    print(json.dumps({"query": query, "ranked_urls": [asdict(item) for item in ranked]}, indent=2))


if __name__ == "__main__":
    main()
