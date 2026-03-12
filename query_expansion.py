import json
import os
from dataclasses import dataclass
from typing import Iterable

from openai import OpenAI
from dotenv import load_dotenv


DEFAULT_MODEL = "gpt-4.1-mini"


load_dotenv()


@dataclass(slots=True)
class QueryExpansionConfig:
    model: str = DEFAULT_MODEL
    expansion_count: int = 5
    temperature: float = 0.4


class QueryExpander:
    def __init__(self, client: OpenAI, config: QueryExpansionConfig | None = None) -> None:
        self.client = client
        self.config = config or QueryExpansionConfig()

    def expand(self, query: str) -> list[str]:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")

        completion = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You expand a user's web search query into multiple distinct search-engine-ready "
                        "queries. Return valid JSON with a single key named 'queries'. "
                        "Each query must target a meaningfully different retrieval angle, avoid near-duplicates, "
                        "preserve the user's intent, and be concise enough to send directly to a search engine."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original query: {cleaned_query}\n"
                        f"Generate exactly {self.config.expansion_count} distinct web search queries.\n"
                        "Output JSON only in this format:\n"
                        '{"queries": ["...", "..."]}'
                    ),
                },
            ],
        )

        message = completion.choices[0].message.content
        if not message:
            raise RuntimeError("OpenAI returned an empty response")

        payload = json.loads(message)
        queries = self._normalize_queries(payload.get("queries", []))

        if len(queries) < self.config.expansion_count:
            raise RuntimeError(
                f"expected {self.config.expansion_count} queries but only received {len(queries)} distinct queries"
            )

        return queries[: self.config.expansion_count]

    @staticmethod
    def _normalize_queries(queries: Iterable[object]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for item in queries:
            if not isinstance(item, str):
                continue
            query = " ".join(item.split()).strip()
            if not query:
                continue
            dedupe_key = query.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(query)

        return normalized


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")
    return OpenAI(api_key=api_key)
