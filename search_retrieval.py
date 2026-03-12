from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


load_dotenv()

DUCKDUCKGO_HTML_SEARCH_URL = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_BROWSER_SEARCH_URL = "https://duckduckgo.com/?q={query}"


@dataclass(slots=True)
class SearchResult:
    provider: str
    query: str
    title: str
    url: str
    snippet: str = ""


class DuckDuckGoSearchClient:
    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            }
        )

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        response = self.session.get(
            DUCKDUCKGO_HTML_SEARCH_URL,
            params={"q": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_results(query=query, html=response.text, limit=limit)

    def _parse_results(self, query: str, html: str, limit: int) -> list[SearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for result_node in soup.select(".result, .web-result"):
            anchor = result_node.select_one("a.result__a, a.result-link")
            if anchor is None:
                continue
            href = self._normalize_duckduckgo_url(anchor.get("href", "").strip())
            title = " ".join(anchor.get_text(" ", strip=True).split())
            snippet_node = result_node.select_one(".result__snippet, .result-snippet")
            snippet = ""
            if snippet_node is not None:
                snippet = " ".join(snippet_node.get_text(" ", strip=True).split())
            if not href or not title or href in seen_urls:
                continue
            seen_urls.add(href)
            results.append(
                SearchResult(
                    provider="duckduckgo",
                    query=query,
                    title=title,
                    url=href,
                    snippet=snippet,
                )
            )
            if len(results) == limit:
                break

        return results

    @staticmethod
    def _normalize_duckduckgo_url(url: str) -> str:
        if not url:
            return url

        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        redirect_target = query_params.get("uddg")
        if redirect_target:
            return unquote(redirect_target[0])
        return url


class FirefoxDuckDuckGoSearchClient:
    def __init__(
        self,
        timeout: float = 15.0,
        headless: bool = True,
        geckodriver_path: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.headless = headless
        self.geckodriver_path = geckodriver_path

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        options = FirefoxOptions()
        if self.headless:
            options.add_argument("-headless")

        service = FirefoxService(executable_path=self.geckodriver_path) if self.geckodriver_path else None
        driver = webdriver.Firefox(options=options, service=service)

        try:
            driver.get(DUCKDUCKGO_BROWSER_SEARCH_URL.format(query=quote_plus(query)))
            self._wait_for_results(driver)
            return self._extract_results(driver=driver, query=query, limit=limit)
        finally:
            driver.quit()

    def _wait_for_results(self, driver: webdriver.Firefox) -> None:
        wait = WebDriverWait(driver, self.timeout)
        try:
            wait.until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        '[data-testid="result-title-a"], a.result__a, article[data-testid="result"] a',
                    )
                )
            )
        except TimeoutException as exc:
            raise RuntimeError("Firefox search timed out waiting for DuckDuckGo results") from exc

    def _extract_results(self, driver: webdriver.Firefox, query: str, limit: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        anchors = driver.find_elements(
            By.CSS_SELECTOR,
            '[data-testid="result-title-a"], a.result__a, article[data-testid="result"] a',
        )

        for anchor in anchors:
            href = DuckDuckGoSearchClient._normalize_duckduckgo_url((anchor.get_attribute("href") or "").strip())
            title = " ".join(anchor.text.split())
            snippet = self._extract_snippet(anchor)
            if not href or not title or href in seen_urls:
                continue
            seen_urls.add(href)
            results.append(
                SearchResult(
                    provider="firefox",
                    query=query,
                    title=title,
                    url=href,
                    snippet=snippet,
                )
            )
            if len(results) == limit:
                break

        return results

    @staticmethod
    def _extract_snippet(anchor) -> str:
        selectors = [
            "ancestor::article[1]//*[contains(@data-result, 'snippet')]",
            "ancestor::article[1]//*[contains(@class, 'snippet')]",
            "ancestor::div[contains(@class, 'result')][1]//*[contains(@class, 'snippet')]",
        ]
        for selector in selectors:
            elements = anchor.find_elements(By.XPATH, selector)
            for element in elements:
                text = " ".join(element.text.split())
                if text:
                    return text
        return ""


class SearchRetriever:
    def __init__(
        self,
        duckduckgo_client: DuckDuckGoSearchClient | None = None,
        firefox_client: FirefoxDuckDuckGoSearchClient | None = None,
    ) -> None:
        self.duckduckgo_client = duckduckgo_client or DuckDuckGoSearchClient()
        self.firefox_client = firefox_client or FirefoxDuckDuckGoSearchClient()

    def search_queries(self, queries: Iterable[str], limit_per_provider: int = 5) -> dict[str, dict[str, list[dict[str, str]]]]:
        payload: dict[str, dict[str, list[dict[str, str]]]] = {}

        for query in queries:
            payload[query] = {
                "duckduckgo": [asdict(result) for result in self.duckduckgo_client.search(query, limit_per_provider)],
                "firefox": [asdict(result) for result in self.firefox_client.search(query, limit_per_provider)],
            }

        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top URLs for one or more search queries.")
    parser.add_argument("queries", nargs="+", help="One or more search queries to execute.")
    parser.add_argument(
        "--results-per-provider",
        type=int,
        default=5,
        help="Number of URLs to retrieve per provider for each query. Default: 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = SearchRetriever()
    payload = retriever.search_queries(
        queries=args.queries,
        limit_per_provider=args.results_per_provider,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
