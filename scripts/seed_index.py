"""Offline indexing (R8): load seed data into MySQL and ChromaDB.

Usage::

    python -m scripts.seed_index
    python -m scripts.seed_index --tickets data/tickets.json --knowledge data/knowledge.json

Text is redacted before it is stored or embedded, as in the online path. The script
is safe to re-run: records are replaced by id. Only approved knowledge is embedded.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.cache_and_security import redact_text, ticket_document
from app.config import get_settings
from app.database import KnowledgeArticle, Repository, Ticket, VectorStore
from app.embeddings import Embedder, build_embedder
from app.schemas import AccessLevel


def _redacted(value: str | None) -> str | None:
    return redact_text(value)[0] if value else value


def _ticket_values(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert one seed record to redacted column values."""
    return {
        "ticket_id": raw["ticket_id"],
        "title": _redacted(raw["title"])[:255],
        "description": _redacted(raw["description"]),
        "symptoms": _redacted(raw.get("symptoms")),
        "affected_service": raw["affected_service"],
        "environment": (raw.get("environment") or "").lower() or None,
        "category": raw.get("category"),
        "subcategory": raw.get("subcategory"),
        "priority": raw.get("priority"),
        "status": raw["status"],
        "assignment_group": raw.get("assignment_group"),
        "resolution_notes": _redacted(raw.get("resolution_notes")),
        "resolution_code": raw.get("resolution_code"),
        "access_rank": AccessLevel(raw.get("access_level", "INTERNAL")).rank,
        "created_at": datetime.fromisoformat(raw["created_at"]),
        "resolved_at": datetime.fromisoformat(raw["resolved_at"]) if raw.get("resolved_at") else None,
    }


def _article_values(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert one knowledge record to redacted column values."""
    return {
        "article_id": raw["article_id"],
        "title": _redacted(raw["title"])[:255],
        "content": _redacted(raw["content"]),
        "affected_service": raw.get("affected_service"),
        "approved": raw.get("approved", False),
        "access_rank": AccessLevel(raw.get("access_level", "INTERNAL")).rank,
    }


def seed(
    repo: Repository,
    vectors: VectorStore,
    embedder: Embedder,
    tickets_path: Path,
    knowledge_path: Path,
) -> tuple[int, int]:
    """Load both files and return the number of tickets and articles indexed."""
    repo.create_schema()

    tickets = json.loads(tickets_path.read_text(encoding="utf-8"))["tickets"]
    for raw in tickets:
        values = _ticket_values(raw)
        repo.upsert_seed_ticket(values)
        document = ticket_document(values)
        vectors.index_ticket(Ticket(**values), document, embedder.embed(document))

    articles = json.loads(knowledge_path.read_text(encoding="utf-8"))["articles"]
    indexed_articles = 0
    for raw in articles:
        values = _article_values(raw)
        repo.upsert_seed_article(values)
        if values["approved"]:
            document = f"{values['title']}\n{values['content']}"
            vectors.index_article(KnowledgeArticle(**values), document, embedder.embed(document))
            indexed_articles += 1
    return len(tickets), indexed_articles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickets", type=Path, default=Path("data/tickets.json"))
    parser.add_argument("--knowledge", type=Path, default=Path("data/knowledge.json"))
    args = parser.parse_args()

    settings = get_settings()
    tickets, articles = seed(
        Repository(settings.database_url),
        VectorStore(settings),
        build_embedder(settings),
        args.tickets,
        args.knowledge,
    )
    print(f"Indexed {tickets} tickets and {articles} approved knowledge articles")


if __name__ == "__main__":
    main()
