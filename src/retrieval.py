"""Local RAG over the policy knowledge base.

Pipeline: load .md policies -> split into one chunk per numbered rule ->
embed with Gemini -> cache vectors to a local .npz -> cosine top-k at query time.

Chunking is per numbered rule rather than by character window because the
policies are already written as short, self-contained clauses ("Damage must be
reported within 7 calendar days of delivery."). Splitting on that natural
boundary keeps each chunk atomic, so a retrieved chunk is a complete rule and
`sources` can point at the exact file it came from.

Only the embedding *call* is remote; the vectors and the search are local, and
the index is rebuilt automatically when the knowledge base changes.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)

# Guards one-time construction of the shared client and the index.
_init_lock = threading.RLock()

# "1. Some rule text" at the start of a line, running until the next such marker.
_RULE_RE = re.compile(r"^\s*(\d+)\.\s+(.*?)(?=^\s*\d+\.\s+|\Z)", re.MULTILINE | re.DOTALL)
_HEADING_RE = re.compile(r"^#\s*(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str  # filename, e.g. "damaged_goods.md"
    rule: str    # e.g. "2"

    @property
    def citation(self) -> str:
        return f"{self.source}#rule-{self.rule}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


# --------------------------------------------------------------------------
# Loading + chunking
# --------------------------------------------------------------------------


def load_chunks(kb_dir: Path | None = None) -> list[Chunk]:
    """Read every policy file and split it into one chunk per numbered rule."""
    kb_dir = kb_dir or settings.knowledge_base_dir
    chunks: list[Chunk] = []

    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        heading_match = _HEADING_RE.search(raw)
        heading = heading_match.group(1).strip() if heading_match else path.stem

        for rule_no, body in _RULE_RE.findall(raw):
            body = " ".join(body.split())
            if not body:
                continue
            # The heading is prepended so the embedding carries topic context;
            # rule 2 of "Damaged Goods" and rule 2 of "Returns" read alike alone.
            chunks.append(
                Chunk(
                    text=f"{heading} (rule {rule_no}): {body}",
                    source=path.name,
                    rule=rule_no,
                )
            )

    if not chunks:
        raise RuntimeError(f"No policy rules found in {kb_dir}")
    return chunks


def kb_fingerprint(chunks: list[Chunk]) -> str:
    """Hash of the chunk texts, used to detect a stale cached index."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.citation.encode())
        h.update(c.text.encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _make_client():
    """Created lazily so importing this module never requires an API key."""
    from google import genai

    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=settings.gemini_api_key)


def get_client():
    """The one shared Gemini client.

    Lock-guarded because `lru_cache` alone does not stop two threads entering
    the factory at once. When that happened, the surplus clients were garbage
    collected and closed the HTTP transport out from under the cached one
    ("Cannot send a request, as the client has been closed").
    """
    with _init_lock:
        return _make_client()


def embed(texts: list[str], *, task_type: str) -> np.ndarray:
    """Embed texts and return L2-normalised row vectors.

    Normalising here means cosine similarity is just a dot product later.
    """
    from google.genai import types

    response = get_client().models.embed_content(
        model=settings.embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    vectors = np.array([e.values for e in response.embeddings], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


# --------------------------------------------------------------------------
# Index build / load
# --------------------------------------------------------------------------


class PolicyIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks = chunks
        self.vectors = vectors

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieval_top_k
        q = embed([query], task_type="RETRIEVAL_QUERY")[0]
        scores = self.vectors @ q  # both sides are unit vectors -> cosine
        best = np.argsort(-scores)[:top_k]
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in best]


def build_index(index_path: Path | None = None, kb_dir: Path | None = None) -> PolicyIndex:
    """Embed the knowledge base and write the vectors to disk."""
    index_path = index_path or settings.index_path
    chunks = load_chunks(kb_dir)
    logger.info("Embedding %d policy rules", len(chunks))

    vectors = embed([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        index_path,
        vectors=vectors,
        texts=np.array([c.text for c in chunks]),
        sources=np.array([c.source for c in chunks]),
        rules=np.array([c.rule for c in chunks]),
        fingerprint=np.array(kb_fingerprint(chunks)),
    )
    return PolicyIndex(chunks, vectors)


@lru_cache(maxsize=1)
def _load_index() -> PolicyIndex:
    """Load the cached index, rebuilding it if missing or stale."""
    index_path = settings.index_path
    current = load_chunks()

    if index_path.exists():
        data = np.load(index_path, allow_pickle=False)
        if str(data["fingerprint"]) == kb_fingerprint(current):
            chunks = [
                Chunk(text=str(t), source=str(s), rule=str(r))
                for t, s, r in zip(data["texts"], data["sources"], data["rules"], strict=True)
            ]
            return PolicyIndex(chunks, data["vectors"])
        logger.info("Knowledge base changed - rebuilding index")

    return build_index()


def get_index() -> PolicyIndex:
    """Thread-safe accessor: only one thread may build or load the index."""
    with _init_lock:
        return _load_index()


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    return get_index().search(query, top_k)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    logging.basicConfig(level=logging.INFO)
    found = load_chunks()
    print(f"{len(found)} chunks from {len({c.source for c in found})} policy files")
    for c in found[:3]:
        print(f"  [{c.citation}] {c.text[:80]}...")
