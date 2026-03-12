from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from html import unescape
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from readability import Document
except ImportError:
    Document = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


load_dotenv()

DEFAULT_TIMEOUT = 20.0
DEFAULT_CACHE_TTL_SECONDS = 900
DEFAULT_RATE_LIMIT_RPS = 1.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
CLOUDFLARE_CONTENT_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/browser-rendering/content"
VISION_PROMPT = (
    "Extract the most important readable content from this webpage screenshot. "
    "Return a concise plain text summary that preserves visible headings, key values, "
    "tables, and bullet points when possible."
)
BOT_PROTECTION_PATTERNS = {
    "cloudflare": (
        "cf-browser-verification",
        "checking your browser",
        "cloudflare ray id",
        "attention required!",
        "__cf_chl_tk",
        "cf-challenge",
    ),
    "captcha": (
        "captcha",
        "g-recaptcha",
        "hcaptcha",
        "verify you are human",
        "press and hold",
    ),
    "datadome": (
        "datadome",
        "ddos protection by datadome",
    ),
    "akamai": (
        "akamai",
        "akamai bot manager",
        "abck",
        "bm_sz",
    ),
    "imperva": (
        "imperva",
        "incapsula",
        "_incap_",
    ),
}
LOW_SIGNAL_PATTERNS = (
    "sign in",
    "log in",
    "create account",
    "accept cookies",
    "cookie preferences",
    "subscribe now",
    "start free trial",
    "continue reading",
    "newsletter",
)
PAYWALL_PATTERNS = (
    "subscribe to continue",
    "remaining article",
    "members only",
    "subscriber-only",
    "sign in to continue reading",
)
API_HINT_PATTERN = re.compile(r"(?i)(/api/[^\"'\\\s<>]+|/v[12]/[^\"'\\\s<>]+|/graphql[^\"'\\\s<>]*)")
ABSOLUTE_API_PATTERN = re.compile(
    r"""(?ix)
    https?://[^\s"'<>]+?(?:/api/|/v1/|/v2/|/graphql)[^\s"'<>]*
    """
)
FETCH_CALL_PATTERN = re.compile(
    r"""(?ix)
    (?:fetch|axios\.(?:get|post|request|put)|XMLHttpRequest)
    [^"'`]{0,120}
    ["'`](?P<url>[^"'`]+)["'`]
    """
)
JSON_ASSIGNMENT_PATTERNS = (
    re.compile(r"window\.__DATA__\s*=\s*", re.I),
    re.compile(r"window\.__INITIAL_STATE__\s*=\s*", re.I),
    re.compile(r"window\.__NUXT__\s*=\s*", re.I),
    re.compile(r"window\.__NEXT_DATA__\s*=\s*", re.I),
    re.compile(r"__APOLLO_STATE__\s*=\s*", re.I),
)


@dataclass(slots=True)
class PageMetadata:
    canonical_url: str = ""
    publish_date: str = ""
    author: str = ""
    description: str = ""
    site_name: str = ""
    content_type: str = ""
    language: str = ""


@dataclass(slots=True)
class TableData:
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""


@dataclass(slots=True)
class LinkData:
    text: str
    url: str
    kind: str = ""


@dataclass(slots=True)
class ImageData:
    url: str
    alt: str = ""


@dataclass(slots=True)
class StructuredDataItem:
    kind: str
    source: str
    data: Any


@dataclass(slots=True)
class ApiEndpoint:
    url: str
    source: str


@dataclass(slots=True)
class FetchMetrics:
    fetch_time_ms: float = 0.0
    content_size_bytes: int = 0
    extraction_score: float = 0.0
    fallback_method_used: str = ""
    cache_hit: bool = False
    rate_limited_ms: float = 0.0
    stage_timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class FetchResult:
    url: str
    final_url: str
    title: str
    content: str
    tables: list[TableData]
    links: list[LinkData]
    images: list[ImageData]
    metadata: PageMetadata
    structured_data: list[StructuredDataItem]
    apis: list[ApiEndpoint]
    raw_html: str
    parsed_html: str
    parsed_text: str
    headings: list[str]
    lists: list[list[str]]
    method: str
    fallback_chain: list[str]
    success: bool
    status_code: int | None = None
    screenshot_path: str = ""
    vision_summary: str = ""
    error: str = ""
    challenge_detected: bool = False
    dynamic_detected: bool = False
    paywall_detected: bool = False
    low_signal_detected: bool = False
    extraction_warnings: list[str] = field(default_factory=list)
    metrics: FetchMetrics = field(default_factory=FetchMetrics)

    def to_public_dict(self) -> dict[str, Any]:
        methods = list(dict.fromkeys(self.fallback_chain or [self.method]))
        public_method: str | list[str] = methods[0] if len(methods) == 1 else methods
        return {
            "url": self.url,
            "final_url": self.final_url,
            "title": self.title,
            "content": self.content,
            "method": public_method,
            "success": self.success,
            "status_code": self.status_code,
        }


@dataclass(slots=True)
class RenderedPage:
    url: str
    final_url: str
    html: str
    title: str
    status_code: int | None
    method: str
    apis: list[ApiEndpoint] = field(default_factory=list)
    screenshot_path: str = ""
    error: str = ""


class RequestThrottle:
    def __init__(self, requests_per_second: float = DEFAULT_RATE_LIMIT_RPS) -> None:
        self.interval = 0.0 if requests_per_second <= 0 else 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed: dict[str, float] = {}

    def wait(self, url: str) -> float:
        if self.interval <= 0:
            return 0.0
        domain = urlparse(url).netloc.lower()
        with self._lock:
            now = time.monotonic()
            allowed_at = self._next_allowed.get(domain, now)
            delay = max(0.0, allowed_at - now)
            self._next_allowed[domain] = max(allowed_at, now) + self.interval
        if delay > 0:
            time.sleep(delay)
        return delay * 1000.0


class FetchCache:
    def __init__(self, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._items: dict[str, tuple[float, FetchResult]] = {}

    @staticmethod
    def _key(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def get(self, url: str) -> FetchResult | None:
        key = self._key(url)
        with self._lock:
            item = self._items.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at < time.time():
                self._items.pop(key, None)
                return None
            return replace(value, metrics=replace(value.metrics, cache_hit=True))

    def set(self, url: str, result: FetchResult) -> None:
        key = self._key(url)
        with self._lock:
            self._items[key] = (time.time() + self.ttl_seconds, result)


class ExtractionUtils:
    @staticmethod
    def build_clean_soup(html: str) -> BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
            tag.decompose()
        for selector in ("header", "footer", "nav", "aside"):
            for node in soup.select(selector):
                node.decompose()
        for selector in (
            ".modal",
            ".popup",
            ".cookie",
            ".advertisement",
            ".ad",
            ".banner",
            ".newsletter",
            ".subscribe",
            ".offcanvas",
            ".dropdown-menu",
            ".navbar",
            ".userOptionsMenu",
            "#onetrust-consent-sdk",
            ".toast",
            ".paywall",
            ".login-wall",
        ):
            for node in soup.select(selector):
                node.decompose()
        for node in soup.select("[style]"):
            attrs = getattr(node, "attrs", None)
            if attrs is None:
                continue
            style = str(attrs.get("style", "")).lower().replace(" ", "")
            if "display:none" in style or "visibility:hidden" in style:
                node.decompose()
                continue
            if "height:0" in style and "width:0" in style:
                node.decompose()
        for node in soup.select("[hidden], [aria-hidden='true'], input[type='hidden']"):
            if getattr(node, "attrs", None) is not None:
                node.decompose()
        return soup

    @staticmethod
    def node_text(node: Tag | BeautifulSoup) -> str:
        return " ".join(node.get_text(" ", strip=True).replace("\ufeff", "").split()).strip()

    @staticmethod
    def parse_raw_html(html: str) -> str:
        return ExtractionUtils.build_clean_soup(html).prettify()

    @staticmethod
    def parse_raw_text(html: str) -> str:
        soup = ExtractionUtils.build_clean_soup(html)
        return " ".join(soup.get_text(" ", strip=True).split())

    @staticmethod
    def text_html_ratio(text: str, html: str) -> float:
        if not html:
            return 0.0
        return len(text.strip()) / max(len(html), 1)

    @staticmethod
    def detect_bot_protection(html: str, status_code: int | None = None) -> tuple[bool, list[str]]:
        lowered = html.lower()
        hits: list[str] = []
        for provider, terms in BOT_PROTECTION_PATTERNS.items():
            if any(term in lowered for term in terms):
                hits.append(provider)
        if status_code in (403, 429) and hits:
            return True, hits
        return bool(hits), hits

    @staticmethod
    def detect_dynamic_page(html: str, parsed_text: str) -> bool:
        lowered = html.lower()
        script_blocks = len(re.findall(r"<script\b", lowered))
        avg_script_size = len(html) / max(script_blocks, 1)
        has_runtime_calls = any(token in lowered for token in ("fetch(", "axios.", "xmlhttprequest", "__next_data__", "__nuxt__"))
        sparse_dom = len(parsed_text) < 400 and script_blocks >= 8
        heavy_scripts = script_blocks >= 5 and avg_script_size > 5000
        return sparse_dom or heavy_scripts or has_runtime_calls

    @staticmethod
    def detect_low_signal(text: str, html: str) -> bool:
        lowered = text.lower()
        if any(term in lowered for term in LOW_SIGNAL_PATTERNS):
            return True
        return ExtractionUtils.text_html_ratio(text, html) < 0.01

    @staticmethod
    def detect_paywall(text: str, html: str) -> bool:
        lowered = f"{text} {html}".lower()
        return any(term in lowered for term in PAYWALL_PATTERNS)

    @staticmethod
    def extract_metadata(soup: BeautifulSoup, base_url: str, content_type: str) -> PageMetadata:
        def meta(*selectors: str, attr: str = "content") -> str:
            for selector in selectors:
                tag = soup.select_one(selector)
                if tag:
                    value = (tag.get(attr) or "").strip()
                    if value:
                        return value
            return ""

        canonical = meta("link[rel='canonical']", attr="href")
        description = meta("meta[name='description']", "meta[property='og:description']")
        site_name = meta("meta[property='og:site_name']", "meta[name='application-name']")
        author = meta("meta[name='author']", "meta[property='article:author']")
        publish_date = meta(
            "meta[property='article:published_time']",
            "meta[name='pubdate']",
            "meta[name='publish-date']",
            "meta[itemprop='datePublished']",
        )
        language = (soup.html.get("lang") or "").strip() if soup.html else ""
        return PageMetadata(
            canonical_url=urljoin(base_url, canonical) if canonical else "",
            publish_date=publish_date,
            author=author,
            description=description,
            site_name=site_name,
            content_type=content_type,
            language=language,
        )

    @staticmethod
    def extract_structured_data(html: str, soup: BeautifulSoup) -> list[StructuredDataItem]:
        items: list[StructuredDataItem] = []

        for script in soup.select("script[type='application/ld+json']"):
            content = (script.string or script.get_text() or "").strip()
            if not content:
                continue
            try:
                items.append(StructuredDataItem(kind="ld+json", source="script", data=json.loads(content)))
            except json.JSONDecodeError:
                continue

        for name, pattern in (
            ("window.__DATA__", JSON_ASSIGNMENT_PATTERNS[0]),
            ("window.__INITIAL_STATE__", JSON_ASSIGNMENT_PATTERNS[1]),
            ("window.__NUXT__", JSON_ASSIGNMENT_PATTERNS[2]),
            ("window.__NEXT_DATA__", JSON_ASSIGNMENT_PATTERNS[3]),
            ("__APOLLO_STATE__", JSON_ASSIGNMENT_PATTERNS[4]),
        ):
            for value in ExtractionUtils._extract_json_assignments(html, pattern):
                items.append(StructuredDataItem(kind=name, source="script", data=value))

        for script in soup.select("script"):
            content = (script.string or script.get_text() or "").strip()
            if not content or len(content) < 50:
                continue
            embedded = ExtractionUtils._extract_embedded_json(content)
            for value in embedded[:2]:
                items.append(StructuredDataItem(kind="embedded_json", source="script", data=value))

        return items

    @staticmethod
    def _extract_json_assignments(text: str, pattern: re.Pattern[str]) -> list[Any]:
        values: list[Any] = []
        for match in pattern.finditer(text):
            payload = ExtractionUtils._read_balanced_json(text, match.end())
            if payload is None:
                continue
            try:
                values.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
        return values

    @staticmethod
    def _extract_embedded_json(text: str) -> list[Any]:
        values: list[Any] = []
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                values.append(json.loads(stripped))
            except json.JSONDecodeError:
                pass

        for token in ('{"', "[{"):
            start = stripped.find(token)
            if start < 0:
                continue
            payload = ExtractionUtils._read_balanced_json(stripped, start)
            if payload is None:
                continue
            try:
                values.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
        return values

    @staticmethod
    def _read_balanced_json(text: str, start_index: int) -> str | None:
        if start_index >= len(text):
            return None
        while start_index < len(text) and text[start_index].isspace():
            start_index += 1
        if start_index >= len(text) or text[start_index] not in "[{":
            return None
        opening = text[start_index]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for index in range(start_index, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start_index : index + 1]
        return None

    @staticmethod
    def discover_api_endpoints(html: str, base_url: str) -> list[ApiEndpoint]:
        seen: dict[str, ApiEndpoint] = {}
        for match in ABSOLUTE_API_PATTERN.findall(html):
            seen[match] = ApiEndpoint(url=match, source="html")
        for match in API_HINT_PATTERN.findall(html):
            absolute = urljoin(base_url, match)
            seen[absolute] = ApiEndpoint(url=absolute, source="html")
        for match in FETCH_CALL_PATTERN.finditer(html):
            candidate = match.group("url").strip()
            if any(hint in candidate.lower() for hint in ("/api/", "/v1/", "/v2/", "/graphql")):
                absolute = urljoin(base_url, candidate)
                seen[absolute] = ApiEndpoint(url=absolute, source="script")
        return list(seen.values())

    @staticmethod
    def extract_tables(soup: BeautifulSoup) -> list[TableData]:
        tables: list[TableData] = []
        for table in soup.select("table")[:12]:
            caption_node = table.find("caption")
            caption = ExtractionUtils.node_text(caption_node) if caption_node else ""
            rows = table.select("tr")
            parsed_rows: list[list[str]] = []
            for row in rows:
                cells = [ExtractionUtils.node_text(cell) for cell in row.select("th, td")]
                cells = [cell for cell in cells if cell]
                if cells:
                    parsed_rows.append(cells)
            if not parsed_rows:
                continue
            headers = parsed_rows[0] if table.select("th") else []
            data_rows = parsed_rows[1:] if headers else parsed_rows
            tables.append(TableData(headers=headers, rows=data_rows[:50], caption=caption))
        return tables

    @staticmethod
    def extract_links(soup: BeautifulSoup, base_url: str) -> list[LinkData]:
        links: list[LinkData] = []
        seen: set[str] = set()
        for anchor in soup.select("a[href]")[:200]:
            href = (anchor.get("href") or "").strip()
            if not href or href.startswith(("javascript:", "mailto:", "#")):
                continue
            absolute = urljoin(base_url, href)
            if absolute in seen:
                continue
            seen.add(absolute)
            rel = " ".join(anchor.get("rel", []))
            text = ExtractionUtils.node_text(anchor)
            kind = "reference" if "nofollow" not in rel else "nofollow"
            links.append(LinkData(text=text or absolute, url=absolute, kind=kind))
        return links

    @staticmethod
    def extract_images(soup: BeautifulSoup, base_url: str) -> list[ImageData]:
        images: list[ImageData] = []
        seen: set[str] = set()
        for image in soup.select("img[src]")[:100]:
            src = (image.get("src") or "").strip()
            if not src:
                continue
            absolute = urljoin(base_url, src)
            if absolute in seen:
                continue
            seen.add(absolute)
            images.append(ImageData(url=absolute, alt=(image.get("alt") or "").strip()))
        return images

    @staticmethod
    def extract_headings(soup: BeautifulSoup) -> list[str]:
        headings: list[str] = []
        for node in soup.select("h1, h2, h3")[:40]:
            text = ExtractionUtils.node_text(node)
            if text:
                headings.append(text)
        return headings

    @staticmethod
    def extract_lists(soup: BeautifulSoup) -> list[list[str]]:
        result: list[list[str]] = []
        for list_node in soup.select("ul, ol")[:20]:
            items = [ExtractionUtils.node_text(item) for item in list_node.select("li")]
            items = [item for item in items if item]
            if len(items) >= 2:
                result.append(items[:25])
        return result

    @staticmethod
    def extract_content(html: str, base_url: str, structured_data: list[StructuredDataItem]) -> tuple[str, str, float, list[str]]:
        warnings: list[str] = []
        soup = ExtractionUtils.build_clean_soup(html)
        title = ExtractionUtils._extract_title(soup)

        content_candidates: list[tuple[str, float]] = []
        parsed_text = ExtractionUtils.parse_raw_text(html)
        if parsed_text:
            content_candidates.append((parsed_text, ExtractionUtils._score_text(parsed_text, soup)))

        if trafilatura is not None:
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_links=False,
                include_tables=True,
                favor_precision=True,
            )
            if extracted:
                text = " ".join(extracted.split())
                content_candidates.append((text, ExtractionUtils._score_text(text, soup) + 120))
        else:
            warnings.append("trafilatura is not installed")

        if Document is not None:
            doc = Document(html)
            if not title:
                title = " ".join((doc.short_title() or "").split())
            summary_html = doc.summary(html_partial=True)
            summary_text = ExtractionUtils.node_text(BeautifulSoup(summary_html, "html.parser"))
            if summary_text:
                content_candidates.append((summary_text, ExtractionUtils._score_text(summary_text, soup) + 80))
        else:
            warnings.append("readability-lxml is not installed")

        domain_specific = ExtractionUtils._extract_domain_specific(base_url, html)
        if domain_specific:
            content_candidates.append((domain_specific, ExtractionUtils._score_text(domain_specific, soup) + 140))

        for item in structured_data:
            candidate = ExtractionUtils._summarize_structured_data(item.data)
            if candidate:
                content_candidates.append((candidate, len(candidate)))

        if not content_candidates:
            return title, "", 0.0, warnings

        content, score = max(content_candidates, key=lambda item: item[1])
        return title, content, score, warnings

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            return " ".join(soup.title.string.split())
        og_title = soup.select_one("meta[property='og:title']")
        if og_title:
            return " ".join((og_title.get("content") or "").split())
        h1 = soup.select_one("h1")
        return ExtractionUtils.node_text(h1) if h1 else ""

    @staticmethod
    def _summarize_structured_data(value: Any) -> str:
        if isinstance(value, dict):
            parts: list[str] = []
            for key in ("headline", "name", "description", "articleBody", "text"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    parts.append(candidate.strip())
            return " ".join(parts)
        if isinstance(value, list):
            parts = []
            for item in value[:5]:
                summary = ExtractionUtils._summarize_structured_data(item)
                if summary:
                    parts.append(summary)
            return " ".join(parts)
        return ""

    @staticmethod
    def _score_text(text: str, soup: BeautifulSoup) -> float:
        cleaned = " ".join(text.split())
        alpha_count = sum(1 for char in cleaned if char.isalpha())
        digit_count = sum(1 for char in cleaned if char.isdigit())
        unique_tokens = {token.lower() for token in cleaned.split() if len(token) > 2}
        paragraph_count = len(soup.select("p"))
        heading_count = len(soup.select("h1, h2, h3"))
        table_count = len(soup.select("table"))
        return (
            len(cleaned)
            + alpha_count * 0.25
            + digit_count * 0.4
            + len(unique_tokens) * 3.0
            + paragraph_count * 20.0
            + heading_count * 12.0
            + table_count * 18.0
        )

    @staticmethod
    def _extract_domain_specific(base_url: str, html: str) -> str:
        domain = urlparse(base_url).netloc.lower()
        soup = ExtractionUtils.build_clean_soup(html)

        if "wikipedia.org" in domain:
            body = soup.select_one("#mw-content-text")
            return ExtractionUtils.node_text(body) if body else ""

        if "tradingeconomics.com" in domain:
            nodes = soup.select(".table, .table-responsive, .historical-data, .datatable, .card")
            return " ".join(ExtractionUtils.node_text(node) for node in nodes[:8])

        if "finance.yahoo.com" in domain:
            nodes = soup.select('[data-test="qsp-statistics"], [data-test="quote-summary"], table, section')
            return " ".join(ExtractionUtils.node_text(node) for node in nodes[:8])

        if "docs.github.com" in domain:
            article = soup.select_one("main")
            return ExtractionUtils.node_text(article) if article else ""

        if "stackoverflow.com" in domain:
            nodes = soup.select(".question, .answer, .js-post-body")
            return " ".join(ExtractionUtils.node_text(node) for node in nodes[:6])

        return ""

    @staticmethod
    def is_meaningful_content(text: str, title: str = "") -> bool:
        cleaned = " ".join(text.split()).strip()
        if len(cleaned) < 120:
            return False
        lowered = cleaned.lower()
        if any(term in lowered for term in ("enable javascript", "access denied", "checking your browser")):
            return False
        alpha_count = sum(1 for char in cleaned if char.isalpha())
        digit_count = sum(1 for char in cleaned if char.isdigit())
        if alpha_count < 80 and digit_count < 20:
            return False
        unique_tokens = {token.lower() for token in cleaned.split() if len(token) > 2}
        if len(unique_tokens) >= 20:
            return True
        title_lower = title.lower()
        data_page_hints = ("market", "quote", "chart", "historical data", "index", "stocks", "bond", "fx")
        return digit_count >= 20 and any(hint in title_lower for hint in data_page_hints)


class HttpContentFetcher:
    def __init__(self, timeout: float = DEFAULT_TIMEOUT, throttle: RequestThrottle | None = None) -> None:
        self.timeout = timeout
        self.throttle = throttle or RequestThrottle()
        self.session = requests.Session()
        retry_config = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry_config))
        self.session.mount("https://", HTTPAdapter(max_retries=retry_config))
        self.session.headers.update(
            {
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Dest": "document",
            }
        )

    def fetch(self, url: str) -> tuple[requests.Response | None, float, str]:
        rate_limited_ms = self.throttle.wait(url)
        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            return response, rate_limited_ms, ""
        except requests.RequestException as exc:
            return None, rate_limited_ms, str(exc)


class CloudflareBrowserContentFetcher:
    def __init__(
        self,
        account_id: str | None = None,
        api_token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.api_token = api_token or os.getenv("CLOUDFLARE_API_TOKEN")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_token}" if self.api_token else "",
                "Content-Type": "application/json",
            }
        )

    def is_configured(self) -> bool:
        return bool(self.account_id and self.api_token)

    def fetch(self, url: str) -> RenderedPage:
        if not self.is_configured():
            return RenderedPage(url=url, final_url=url, html="", title="", status_code=None, method="cloudflare", error="Cloudflare credentials are not configured")
        try:
            response = self.session.post(
                CLOUDFLARE_CONTENT_URL.format(account_id=self.account_id),
                json={
                    "url": url,
                    "userAgent": DEFAULT_USER_AGENT,
                    "gotoOptions": {"waitUntil": "networkidle0"},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            return RenderedPage(url=url, final_url=url, html="", title="", status_code=status_code, method="cloudflare", error=str(exc))

        payload = response.json()
        if not payload.get("success"):
            return RenderedPage(
                url=url,
                final_url=url,
                html="",
                title="",
                status_code=response.status_code,
                method="cloudflare",
                error=f"Cloudflare content fetch failed: {payload}",
            )

        result = payload.get("result", {})
        if isinstance(result, str):
            return RenderedPage(url=url, final_url=url, html=result, title="", status_code=response.status_code, method="cloudflare")
        if isinstance(result, dict):
            return RenderedPage(
                url=url,
                final_url=result.get("url", url),
                html=result.get("content", ""),
                title=result.get("title", ""),
                status_code=response.status_code,
                method="cloudflare",
            )
        return RenderedPage(
            url=url,
            final_url=url,
            html="",
            title="",
            status_code=response.status_code,
            method="cloudflare",
            error=f"Unexpected Cloudflare response shape: {type(result).__name__}",
        )


class PlaywrightBrowserContentFetcher:
    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    def fetch(self, url: str, capture_screenshot: bool = False) -> RenderedPage:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return RenderedPage(url=url, final_url=url, html="", title="", status_code=None, method="playwright", error="playwright is not installed")

        api_hits: dict[str, ApiEndpoint] = {}
        screenshot_path = ""

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(user_agent=DEFAULT_USER_AGENT)

                def on_response(response: Any) -> None:
                    response_url = response.url
                    if any(hint in response_url.lower() for hint in ("/api/", "/v1/", "/v2/", "/graphql")):
                        api_hits[response_url] = ApiEndpoint(url=response_url, source="browser-network")

                page.on("response", on_response)
                response = page.goto(url, wait_until="networkidle", timeout=int(self.timeout * 1000))
                title = page.title()
                html = page.content()
                final_url = page.url
                if capture_screenshot:
                    handle = tempfile.NamedTemporaryFile(prefix="fetch_content_", suffix=".png", delete=False, dir="/tmp")
                    handle.close()
                    page.screenshot(path=handle.name, full_page=True)
                    screenshot_path = handle.name
                browser.close()
                return RenderedPage(
                    url=url,
                    final_url=final_url,
                    html=html,
                    title=title,
                    status_code=response.status if response else None,
                    method="playwright",
                    apis=list(api_hits.values()),
                    screenshot_path=screenshot_path,
                )
        except Exception as exc:  # noqa: BLE001
            return RenderedPage(url=url, final_url=url, html="", title="", status_code=None, method="playwright", error=str(exc), screenshot_path=screenshot_path)


class ScreenshotVisionFallback:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")

    def analyze(self, screenshot_path: str) -> str:
        if not screenshot_path or OpenAI is None or not os.getenv("OPENAI_API_KEY"):
            return ""
        try:
            client = OpenAI()
            with open(screenshot_path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": VISION_PROMPT},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"},
                        ],
                    }
                ],
            )
            return getattr(response, "output_text", "") or ""
        except Exception:  # noqa: BLE001
            return ""


class ContentFetcher:
    def __init__(
        self,
        http_fetcher: HttpContentFetcher | None = None,
        cloudflare_fetcher: CloudflareBrowserContentFetcher | None = None,
        playwright_fetcher: PlaywrightBrowserContentFetcher | None = None,
        vision_fallback: ScreenshotVisionFallback | None = None,
        cache: FetchCache | None = None,
    ) -> None:
        self.http_fetcher = http_fetcher or HttpContentFetcher()
        self.cloudflare_fetcher = cloudflare_fetcher or CloudflareBrowserContentFetcher()
        self.playwright_fetcher = playwright_fetcher or PlaywrightBrowserContentFetcher()
        self.vision_fallback = vision_fallback or ScreenshotVisionFallback()
        self.cache = cache or FetchCache()

    def fetch(self, url: str) -> FetchResult:
        cached = self.cache.get(url)
        if cached is not None:
            return cached

        started_at = time.perf_counter()
        stage_timings: dict[str, float] = {}
        fallback_chain: list[str] = ["http"]

        http_started = time.perf_counter()
        response, rate_limited_ms, error = self.http_fetcher.fetch(url)
        stage_timings["http_fetch"] = (time.perf_counter() - http_started) * 1000.0

        if response is None:
            result = FetchResult(
                url=url,
                final_url=url,
                title="",
                content="",
                tables=[],
                links=[],
                images=[],
                metadata=PageMetadata(),
                structured_data=[],
                apis=[],
                raw_html="",
                parsed_html="",
                parsed_text="",
                headings=[],
                lists=[],
                method="http",
                fallback_chain=fallback_chain,
                success=False,
                error=error,
                metrics=FetchMetrics(
                    fetch_time_ms=(time.perf_counter() - started_at) * 1000.0,
                    fallback_method_used="http",
                    rate_limited_ms=rate_limited_ms,
                    stage_timings_ms=stage_timings,
                ),
            )
            self.cache.set(url, result)
            return result

        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type.lower():
            result = FetchResult(
                url=url,
                final_url=str(response.url),
                title="",
                content=response.text.strip(),
                tables=[],
                links=[],
                images=[],
                metadata=PageMetadata(content_type=content_type),
                structured_data=[],
                apis=[],
                raw_html="",
                parsed_html="",
                parsed_text=response.text.strip(),
                headings=[],
                lists=[],
                method="http",
                fallback_chain=fallback_chain,
                success=True,
                status_code=response.status_code,
                metrics=FetchMetrics(
                    fetch_time_ms=(time.perf_counter() - started_at) * 1000.0,
                    content_size_bytes=len(response.content),
                    fallback_method_used="http",
                    rate_limited_ms=rate_limited_ms,
                    stage_timings_ms=stage_timings,
                ),
            )
            self.cache.set(url, result)
            return result

        http_result = self._build_result_from_html(
            html=response.text,
            url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            method="http",
            fallback_chain=fallback_chain,
            content_type=content_type,
            started_at=started_at,
            stage_timings=stage_timings,
            rate_limited_ms=rate_limited_ms,
        )

        if self._should_return_http(http_result):
            self.cache.set(url, http_result)
            return http_result

        rendered_result = self._try_browser_fallbacks(url, http_result, started_at, stage_timings, rate_limited_ms)
        self.cache.set(url, rendered_result)
        return rendered_result

    def fetch_many(self, urls: list[str], max_workers: int = 4) -> list[FetchResult]:
        results: list[FetchResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.fetch, url): url for url in urls}
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def _should_return_http(self, result: FetchResult) -> bool:
        if not result.success:
            return False
        if result.challenge_detected or result.dynamic_detected or result.paywall_detected:
            return False
        return True

    def _try_browser_fallbacks(
        self,
        url: str,
        http_result: FetchResult,
        started_at: float,
        stage_timings: dict[str, float],
        rate_limited_ms: float,
    ) -> FetchResult:
        fallback_chain = list(http_result.fallback_chain)
        best_result = http_result

        cloudflare_started = time.perf_counter()
        cloudflare_page = self.cloudflare_fetcher.fetch(url)
        stage_timings["cloudflare_render"] = (time.perf_counter() - cloudflare_started) * 1000.0
        if cloudflare_page.html:
            fallback_chain.append("cloudflare")
            cloudflare_result = self._build_result_from_html(
                html=cloudflare_page.html,
                url=url,
                final_url=cloudflare_page.final_url,
                status_code=cloudflare_page.status_code,
                method="cloudflare",
                fallback_chain=fallback_chain,
                content_type="text/html",
                started_at=started_at,
                stage_timings=stage_timings,
                rate_limited_ms=rate_limited_ms,
                inherited_apis=cloudflare_page.apis,
                screenshot_path=cloudflare_page.screenshot_path,
                prior_error=http_result.error,
            )
            best_result = self._pick_better_result(best_result, cloudflare_result)
            if cloudflare_result.success and not cloudflare_result.challenge_detected:
                return cloudflare_result

        playwright_started = time.perf_counter()
        playwright_page = self.playwright_fetcher.fetch(url, capture_screenshot=True)
        stage_timings["playwright_render"] = (time.perf_counter() - playwright_started) * 1000.0
        fallback_chain = list(http_result.fallback_chain) + ["playwright"]
        if playwright_page.html:
            playwright_result = self._build_result_from_html(
                html=playwright_page.html,
                url=url,
                final_url=playwright_page.final_url,
                status_code=playwright_page.status_code,
                method="playwright",
                fallback_chain=fallback_chain,
                content_type="text/html",
                started_at=started_at,
                stage_timings=stage_timings,
                rate_limited_ms=rate_limited_ms,
                inherited_apis=playwright_page.apis,
                screenshot_path=playwright_page.screenshot_path,
                prior_error=http_result.error or playwright_page.error,
            )
            if self._needs_vision(playwright_result):
                vision_started = time.perf_counter()
                vision_summary = self.vision_fallback.analyze(playwright_page.screenshot_path)
                stage_timings["vision_fallback"] = (time.perf_counter() - vision_started) * 1000.0
                if vision_summary:
                    playwright_result.vision_summary = vision_summary
                    if not playwright_result.content:
                        playwright_result.content = vision_summary
                        playwright_result.success = True
            best_result = self._pick_better_result(best_result, playwright_result)
            if playwright_result.success or playwright_result.vision_summary:
                return playwright_result

        warnings = list(best_result.extraction_warnings)
        if playwright_page.error and self._is_missing_playwright_browser_error(playwright_page.error):
            warnings.append("Playwright browser executable is missing; run `playwright install`")

        if best_result.success or best_result.content:
            return replace(
                best_result,
                error="",
                fallback_chain=list(dict.fromkeys(best_result.fallback_chain + ["cloudflare", "playwright"])),
                extraction_warnings=warnings,
                metrics=replace(
                    best_result.metrics,
                    fetch_time_ms=(time.perf_counter() - started_at) * 1000.0,
                    fallback_method_used=best_result.method,
                    stage_timings_ms=stage_timings,
                ),
            )

        error = best_result.error or cloudflare_page.error or playwright_page.error or "All fetch strategies failed"
        return replace(
            best_result,
            error=error,
            fallback_chain=list(dict.fromkeys(best_result.fallback_chain + ["cloudflare", "playwright"])),
            extraction_warnings=warnings,
            metrics=replace(
                best_result.metrics,
                fetch_time_ms=(time.perf_counter() - started_at) * 1000.0,
                fallback_method_used="playwright" if playwright_page.error else "cloudflare",
                stage_timings_ms=stage_timings,
            ),
        )

    def _needs_vision(self, result: FetchResult) -> bool:
        return bool(result.screenshot_path and (not result.success or result.paywall_detected or result.low_signal_detected))

    @staticmethod
    def _pick_better_result(current: FetchResult, candidate: FetchResult) -> FetchResult:
        if candidate.success and not current.success:
            return candidate
        if candidate.metrics.extraction_score > current.metrics.extraction_score:
            return candidate
        if len(candidate.content.strip()) > len(current.content.strip()):
            return candidate
        if len(candidate.tables) > len(current.tables):
            return candidate
        return current

    @staticmethod
    def _is_missing_playwright_browser_error(error: str) -> bool:
        lowered = error.lower()
        return "browsertype.launch" in lowered and "executable doesn't exist" in lowered

    def _build_result_from_html(
        self,
        html: str,
        url: str,
        final_url: str,
        status_code: int | None,
        method: str,
        fallback_chain: list[str],
        content_type: str,
        started_at: float,
        stage_timings: dict[str, float],
        rate_limited_ms: float,
        inherited_apis: list[ApiEndpoint] | None = None,
        screenshot_path: str = "",
        prior_error: str = "",
    ) -> FetchResult:
        structured_started = time.perf_counter()
        raw_soup = BeautifulSoup(html, "html.parser")
        structured_data = ExtractionUtils.extract_structured_data(html, raw_soup)
        stage_timings[f"{method}_structured_data"] = (time.perf_counter() - structured_started) * 1000.0

        api_started = time.perf_counter()
        apis = ExtractionUtils.discover_api_endpoints(html, final_url)
        if inherited_apis:
            seen = {item.url for item in apis}
            for item in inherited_apis:
                if item.url not in seen:
                    apis.append(item)
        stage_timings[f"{method}_api_discovery"] = (time.perf_counter() - api_started) * 1000.0

        extraction_started = time.perf_counter()
        title, content, score, warnings = ExtractionUtils.extract_content(html, final_url, structured_data)
        parsed_html = ExtractionUtils.parse_raw_html(html)
        parsed_text = ExtractionUtils.parse_raw_text(html)
        clean_soup = ExtractionUtils.build_clean_soup(html)
        metadata = ExtractionUtils.extract_metadata(raw_soup, final_url, content_type)
        tables = ExtractionUtils.extract_tables(clean_soup)
        links = ExtractionUtils.extract_links(clean_soup, final_url)
        images = ExtractionUtils.extract_images(raw_soup, final_url)
        headings = ExtractionUtils.extract_headings(clean_soup)
        lists = ExtractionUtils.extract_lists(clean_soup)
        stage_timings[f"{method}_content_extraction"] = (time.perf_counter() - extraction_started) * 1000.0

        challenge_detected, challenge_sources = ExtractionUtils.detect_bot_protection(html, status_code)
        dynamic_detected = ExtractionUtils.detect_dynamic_page(html, parsed_text)
        paywall_detected = ExtractionUtils.detect_paywall(parsed_text, html)
        low_signal_detected = ExtractionUtils.detect_low_signal(parsed_text, html)
        meaningful = ExtractionUtils.is_meaningful_content(content or parsed_text, title=title)

        if challenge_sources:
            warnings.append(f"bot protection detected: {', '.join(challenge_sources)}")
        if paywall_detected:
            warnings.append("paywall indicators detected")
        if low_signal_detected:
            warnings.append("low-signal page indicators detected")

        if not title:
            title = metadata.site_name or ""

        success = meaningful and not challenge_detected
        error = prior_error
        if not success:
            error = error or f"{method} returned weak or blocked content"

        return FetchResult(
            url=url,
            final_url=final_url,
            title=title,
            content=content or parsed_text,
            tables=tables,
            links=links,
            images=images,
            metadata=metadata,
            structured_data=structured_data,
            apis=apis,
            raw_html=html,
            parsed_html=parsed_html,
            parsed_text=parsed_text,
            headings=headings,
            lists=lists,
            method=method,
            fallback_chain=fallback_chain,
            success=success,
            status_code=status_code,
            screenshot_path=screenshot_path,
            error=error,
            challenge_detected=challenge_detected,
            dynamic_detected=dynamic_detected,
            paywall_detected=paywall_detected,
            low_signal_detected=low_signal_detected,
            extraction_warnings=warnings,
            metrics=FetchMetrics(
                fetch_time_ms=(time.perf_counter() - started_at) * 1000.0,
                content_size_bytes=len(html.encode("utf-8")),
                extraction_score=score,
                fallback_method_used=method,
                cache_hit=False,
                rate_limited_ms=rate_limited_ms,
                stage_timings_ms=dict(stage_timings),
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch page content with staged extraction and browser fallbacks.")
    parser.add_argument("url", help="The URL to fetch content from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fetcher = ContentFetcher()
    result = fetcher.fetch(args.url)
    print(json.dumps(result.to_public_dict(), indent=2))


if __name__ == "__main__":
    main()
