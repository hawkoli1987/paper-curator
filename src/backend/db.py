"""Database operations for paper-curator."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector


def get_connection_string() -> str:
    """Get database connection string from environment."""
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://curator:curator@localhost:5432/paper_curator"
    )


@contextmanager
def get_db() -> Generator[psycopg2.extensions.connection, None, None]:
    """Get a database connection with pgvector support."""
    conn = psycopg2.connect(get_connection_string())
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# Papers CRUD
# =============================================================================

def create_paper(
    arxiv_id: str,
    title: str,
    authors: list[str],
    abstract: Optional[str] = None,
    summary: Optional[str] = None,
    pdf_path: Optional[str] = None,
    latex_path: Optional[str] = None,
    pdf_url: Optional[str] = None,
    published_at: Optional[str] = None,
    embedding: Optional[list[float]] = None,
) -> int:
    """Create a paper and return its ID."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO papers (arxiv_id, title, authors, abstract, summary, 
                                    pdf_path, latex_path, pdf_url, published_at, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    authors = EXCLUDED.authors,
                    abstract = EXCLUDED.abstract,
                    summary = COALESCE(EXCLUDED.summary, papers.summary),
                    pdf_path = COALESCE(EXCLUDED.pdf_path, papers.pdf_path),
                    latex_path = COALESCE(EXCLUDED.latex_path, papers.latex_path),
                    pdf_url = COALESCE(EXCLUDED.pdf_url, papers.pdf_url),
                    published_at = COALESCE(EXCLUDED.published_at, papers.published_at),
                    embedding = COALESCE(EXCLUDED.embedding, papers.embedding)
                RETURNING id
                """,
                (arxiv_id, title, authors, abstract, summary, pdf_path, 
                 latex_path, pdf_url, published_at, embedding)
            )
            paper_id = cur.fetchone()[0]
            conn.commit()
            return paper_id


def get_paper_by_arxiv_id(arxiv_id: str) -> Optional[dict[str, Any]]:
    """Get paper by arXiv ID."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM papers WHERE arxiv_id = %s", (arxiv_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_paper_by_id(paper_id: int) -> Optional[dict[str, Any]]:
    """Get paper by ID."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM papers WHERE id = %s", (paper_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def update_paper_embedding(paper_id: int, embedding: list[float]) -> None:
    """Update paper embedding."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE papers SET embedding = %s WHERE id = %s",
                (embedding, paper_id)
            )
            conn.commit()


def update_paper_summary(paper_id: int, summary: str) -> None:
    """Update paper summary."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE papers SET summary = %s WHERE id = %s",
                (summary, paper_id)
            )
            conn.commit()


def find_similar_papers(embedding: list[float], limit: int = 5, exclude_id: Optional[int] = None) -> list[dict[str, Any]]:
    """Find similar papers by embedding using cosine distance."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if exclude_id:
                cur.execute(
                    """
                    SELECT id, arxiv_id, title, authors, abstract, summary,
                           1 - (embedding <=> %s) AS similarity
                    FROM papers
                    WHERE embedding IS NOT NULL AND id != %s
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (embedding, exclude_id, embedding, limit)
                )
            else:
                cur.execute(
                    """
                    SELECT id, arxiv_id, title, authors, abstract, summary,
                           1 - (embedding <=> %s) AS similarity
                    FROM papers
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (embedding, embedding, limit)
                )
            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Tree CRUD
# =============================================================================

def get_tree() -> list[dict[str, Any]]:
    """Get full tree structure."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT tn.*, p.arxiv_id, p.title as paper_title, p.authors, p.summary
                FROM tree_nodes tn
                LEFT JOIN papers p ON tn.paper_id = p.id
                ORDER BY tn.parent_id NULLS FIRST, tn.position
                """
            )
            return [dict(row) for row in cur.fetchall()]


def add_tree_node(
    node_id: str,
    name: str,
    node_type: str,
    parent_id: Optional[str] = None,
    paper_id: Optional[int] = None,
    position: int = 0,
) -> None:
    """Add a node to the tree."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tree_nodes (node_id, name, node_type, parent_id, paper_id, position)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (node_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    parent_id = EXCLUDED.parent_id,
                    paper_id = EXCLUDED.paper_id,
                    position = EXCLUDED.position
                """,
                (node_id, name, node_type, parent_id, paper_id, position)
            )
            conn.commit()


def delete_tree_node(node_id: str) -> None:
    """Delete a tree node and its children."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # First delete children recursively
            cur.execute(
                "DELETE FROM tree_nodes WHERE parent_id = %s",
                (node_id,)
            )
            cur.execute(
                "DELETE FROM tree_nodes WHERE node_id = %s",
                (node_id,)
            )
            conn.commit()


def get_category_paper_count(category_name: str) -> int:
    """Get number of papers in a category."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM tree_nodes
                WHERE parent_id = (SELECT node_id FROM tree_nodes WHERE name = %s AND node_type = 'category')
                AND node_type = 'paper'
                """,
                (category_name,)
            )
            return cur.fetchone()[0]


# =============================================================================
# Repo Cache
# =============================================================================

def cache_repo(
    paper_id: int,
    source: str,
    repo_url: Optional[str],
    repo_name: Optional[str] = None,
    stars: Optional[int] = None,
    is_official: bool = False,
) -> None:
    """Cache a repo lookup result."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO repo_cache (paper_id, source, repo_url, repo_name, stars, is_official)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (paper_id, source, repo_url, repo_name, stars, is_official)
            )
            conn.commit()


def get_cached_repos(paper_id: int) -> list[dict[str, Any]]:
    """Get cached repos for a paper."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM repo_cache WHERE paper_id = %s ORDER BY is_official DESC, stars DESC NULLS LAST",
                (paper_id,)
            )
            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# References Cache
# =============================================================================

def add_reference(
    source_paper_id: int,
    cited_title: str,
    cited_arxiv_id: Optional[str] = None,
    cited_authors: Optional[list[str]] = None,
    cited_year: Optional[int] = None,
    citation_context: Optional[str] = None,
) -> int:
    """Add a reference and return its ID."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO paper_references (source_paper_id, cited_title, cited_arxiv_id, 
                                              cited_authors, cited_year, citation_context)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (source_paper_id, cited_title, cited_arxiv_id, cited_authors, cited_year, citation_context)
            )
            ref_id = cur.fetchone()[0]
            conn.commit()
            return ref_id


def get_references(paper_id: int) -> list[dict[str, Any]]:
    """Get references for a paper."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM paper_references WHERE source_paper_id = %s ORDER BY id",
                (paper_id,)
            )
            return [dict(row) for row in cur.fetchall()]


def update_reference_explanation(ref_id: int, explanation: str) -> None:
    """Update reference explanation (cache LLM result)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE paper_references SET explanation = %s, explained_at = NOW() WHERE id = %s",
                (explanation, ref_id)
            )
            conn.commit()


# =============================================================================
# Similar Papers Cache
# =============================================================================

def cache_similar_paper(
    paper_id: int,
    similar_arxiv_id: Optional[str],
    similar_title: str,
    similarity_score: Optional[float] = None,
    description: Optional[str] = None,
) -> None:
    """Cache a similar paper result."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO similar_papers_cache (paper_id, similar_arxiv_id, similar_title, similarity_score, description)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (paper_id, similar_arxiv_id, similar_title, similarity_score, description)
            )
            conn.commit()


def get_cached_similar_papers(paper_id: int) -> list[dict[str, Any]]:
    """Get cached similar papers."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM similar_papers_cache WHERE paper_id = %s ORDER BY similarity_score DESC NULLS LAST",
                (paper_id,)
            )
            return [dict(row) for row in cur.fetchall()]
