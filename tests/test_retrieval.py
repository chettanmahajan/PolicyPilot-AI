"""Knowledge-base loading, chunking and cosine search.

Chunking runs against the real policy files and needs no API key. The embedding
call is stubbed with a deterministic fake so search can be tested offline.
"""

import numpy as np
import pytest

from src import retrieval
from src.retrieval import Chunk, PolicyIndex, build_index, kb_fingerprint, load_chunks

EXPECTED_FILES = {
    "cancellations.md",
    "damaged_goods.md",
    "defective_products.md",
    "returns.md",
    "shipping.md",
    "wrong_item.md",
}


# --------------------------------------------------------------------------
# Loading and chunking
# --------------------------------------------------------------------------


def test_every_policy_file_is_loaded():
    chunks = load_chunks()
    assert {c.source for c in chunks} == EXPECTED_FILES


def test_each_numbered_rule_becomes_its_own_chunk():
    chunks = load_chunks()

    # cancellations.md has 4 numbered rules; the others have 5 each.
    per_file = {}
    for c in chunks:
        per_file[c.source] = per_file.get(c.source, 0) + 1

    assert per_file["cancellations.md"] == 4
    assert per_file["damaged_goods.md"] == 5
    assert len(chunks) == 29


def test_chunks_carry_their_heading_and_are_not_truncated():
    chunks = load_chunks()
    damaged = {c.rule: c.text for c in chunks if c.source == "damaged_goods.md"}

    assert damaged["3"].startswith("Damaged Goods Policy (rule 3):")
    # The 2,000 threshold must survive chunking - it is what drives REQUEST_PHOTOS.
    assert "2,000" in damaged["3"]
    assert "photographs" in damaged["3"].lower()


def test_citation_identifies_file_and_rule():
    chunk = Chunk(text="x", source="returns.md", rule="4")
    assert chunk.citation == "returns.md#rule-4"


def test_fingerprint_changes_when_content_changes():
    chunks = load_chunks()
    altered = [*chunks[:-1], Chunk(text="different", source="x.md", rule="1")]

    assert kb_fingerprint(chunks) == kb_fingerprint(load_chunks())
    assert kb_fingerprint(chunks) != kb_fingerprint(altered)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


@pytest.fixture
def fake_embed(monkeypatch):
    """Embed text as a bag-of-words vector over a fixed vocabulary.

    Crude, but it is deterministic, offline, and enough for cosine ranking to
    put lexically-overlapping rules first.
    """
    vocab = [
        "damaged", "photographs", "cancel", "dispatched", "return",
        "unopened", "food", "defective", "shipping", "wrong",
    ]

    def _embed(texts, *, task_type):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([float(lowered.count(word)) for word in vocab])
        vectors = np.array(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    monkeypatch.setattr(retrieval, "embed", _embed)
    return _embed


def test_search_ranks_the_relevant_policy_first(fake_embed):
    chunks = load_chunks()
    vectors = fake_embed([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
    index = PolicyIndex(chunks, vectors)

    results = index.search("my order arrived damaged, do you need photographs?", top_k=3)

    assert results[0].chunk.source == "damaged_goods.md"
    assert results[0].score > 0


def test_search_returns_at_most_top_k(fake_embed):
    chunks = load_chunks()
    vectors = fake_embed([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
    index = PolicyIndex(chunks, vectors)

    assert len(index.search("cancel my order", top_k=4)) == 4


# --------------------------------------------------------------------------
# Index caching
# --------------------------------------------------------------------------


def test_index_round_trips_through_disk(tmp_path, fake_embed):
    index_path = tmp_path / "index.npz"
    built = build_index(index_path=index_path)

    assert index_path.exists()

    data = np.load(index_path, allow_pickle=False)
    assert str(data["fingerprint"]) == kb_fingerprint(load_chunks())
    assert data["vectors"].shape[0] == len(built.chunks)
    assert set(data["sources"].tolist()) == EXPECTED_FILES
