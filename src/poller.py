import logging
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import arxiv
import requests
from sympy import re

from embedder import Paper
from interest_model import DB_PATH


logger = logging.getLogger("poller")

ARXIV_RSS_BASE = "https://rss.arxiv.org/rss/"
DEFAULT_CATEGORIES = ["cs.LG", "cs.AI", "cs.CL", "stat.ML"]
DEFAULT_MAX_RESULTS = 50


def _clean(text: str) -> str:
    return " ".join(text.split())

def _extract_pure_id(url:str) -> str:
    '''
    Extracts pure ArXiv ID (e.g., '2401.12345') from full entry URLs or links.
    Guarantees perfectly synchronized paper_ids across API and RSS.
    '''
    match = re.search(r"abs/([^v]+)", url)
    if match:
        return match.group(1).strip("/")
     
    clean_id = url.split("/")[-1]
    return clean_id.split("v")[0]
def get_user_top_keywords(
    user_id: str,
    db_path: Path = DB_PATH,
    top_k: int = 5,
) -> list[str]:
    """
    Fetches the user's top keywords from keyword_events to build poll queries.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """SELECT keyword, SUM(frequency) as freq
               FROM keyword_events WHERE user_id = ?
               GROUP BY keyword ORDER BY freq DESC LIMIT ?""",
            (user_id, top_k)
        ).fetchall()
    return [r[0] for r in rows]


def fetch_via_api(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    since_days: int = 1,
) -> list[Paper]:
    """
    Fetches recent papers from ArXiv API matching query.
    Filters to papers submitted within the last `since_days` days.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        papers = []
        for result in client.results(search):
            if result.published and result.published < cutoff:
                continue
            pure_id = _extract_pure_id(result.entry_id)
            
            papers.append(Paper(
                paper_id=pure_id,
                title=_clean(result.title),
                abstract=_clean(result.summary),
                authors=[str(a) for a in result.authors],
                url=result.pdf_url,
                published=str(result.published.date()),
            ))
        logger.info(f"API: fetched {len(papers)} papers for query '{query}'")
        return papers
    except Exception as e:
        logger.warning(f"ArXiv API failed for query '{query}': {e}")
        return []


def fetch_via_rss(categories: list[str]) -> list[Paper]:
    """
    Fetches today's papers from ArXiv RSS feeds by category.
    Used as fallback when API is unavailable.
    """
    papers = []
    for cat in categories:
        url = f"{ARXIV_RSS_BASE}{cat}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            channel = root.find("channel")
            if channel is None:
                continue
            for item in channel.findall("item"):
                title_el = item.find("title")
                desc_el = item.find("description")
                link_el = item.find("link")
                if title_el is None or desc_el is None:
                    continue
                pure_id = _extract_pure_id(link_el.text if link_el is not None else "")
                raw_abstract = desc_el.text or ""
                abstract = _clean(raw_abstract.split("Abstract:")[-1] if "Abstract:" in raw_abstract else raw_abstract)
                papers.append(Paper(
                    paper_id=pure_id,
                    title=_clean(title_el.text or ""),
                    abstract=abstract,
                    authors=[],
                    url=link_el.text or "",
                    published=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                ))
            logger.info(f"RSS: fetched {len(papers)} papers from {cat}")
        except Exception as e:
            logger.warning(f"RSS fetch failed for {cat}: {e}")

    seen = set()
    deduped = []
    for p in papers:
        if p.paper_id not in seen:
            seen.add(p.paper_id)
            deduped.append(p)
    return deduped
def poll_new_papers(
    user_id: str,
    rss_cache: Optional[list[Paper]] = None,  
    db_path: Path = DB_PATH,
    max_results: int = DEFAULT_MAX_RESULTS,
    since_days: int = 1,
    fallback_categories: list[str] = DEFAULT_CATEGORIES,
) -> list[Paper]:
    keywords = get_user_top_keywords(user_id, db_path=db_path)
    
 
    def get_fallback_papers():
        if rss_cache is not None:
            logger.info(f"Using pre-fetched RSS cache for user {user_id}")
            return rss_cache
        return fetch_via_rss(fallback_categories)

    if not keywords:
        logger.warning(f"No keywords for user {user_id}, using fallback categories")
        return get_fallback_papers()

     
    query = " OR ".join([f'all:"{k}"' for k in keywords])
    papers = fetch_via_api(query, max_results=max_results, since_days=since_days)

    if not papers:
        logger.info(f"API returned no results for {user_id}, switching to RSS fallback")
        papers = get_fallback_papers()

    seen = set()
    deduped = []
    for p in papers:
        if p.paper_id not in seen:
            seen.add(p.paper_id)
            deduped.append(p)

    logger.info(f"poll_new_papers: {len(deduped)} unique papers for user {user_id}")
    return deduped


def poll_all_users(
    db_path: Path = DB_PATH,
    max_results: int = DEFAULT_MAX_RESULTS,
    since_days: int = 1,
    fallback_categories: list[str] = DEFAULT_CATEGORIES,
) -> dict[str, list[Paper]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT user_id FROM users").fetchall()
    user_ids = [r[0] for r in rows]

    results: dict[str, list[Paper]] = {}
    if not user_ids:
        return results

 
    rss_cached_pool = None
    try:
        logger.info("Pre-fetching RSS feed pool for global fallback session...")
        rss_cached_pool = fetch_via_rss(fallback_categories)
    except Exception as e:
        logger.warning(f"Global RSS pre-fetch failed, users will fetch individually if needed: {e}")

    for uid in user_ids:
        try:
            papers = poll_new_papers(
                user_id=uid,
                rss_cache=rss_cached_pool,  
                max_results=max_results,
                since_days=since_days,
                fallback_categories=fallback_categories,
            )
            results[uid] = papers
        except Exception as e:
            logger.error(f"Polling failed for user {uid}: {e}")
            results[uid] = []

    return results
