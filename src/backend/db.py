"""Database operations for paper-curator."""

from __future__ import annotations

import os
import pathlib
from contextlib import contextmanager
from typing import Any, Generator, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

# ---------------------------------------------------------------------------
# Switchable database connection
# ---------------------------------------------------------------------------

# Module-level mutable state: the active database name.
# Defaults come from DATABASE_URL env var fallback, but can be overridden at
# runtime via switch_database().
_db_config: dict[str, str] = {
    "host": os.environ.get("PGHOST", "localhost"),
    "port": os.environ.get("PGPORT", "5432"),
    "user": os.environ.get("PGUSER", "curator"),
    "password": os.environ.get("PGPASSWORD", "curator"),
    "dbname": os.environ.get("PGDATABASE", "paper_curator"),
}


def _load_db_config_from_yaml() -> None:
    """Seed _db_config from config/config.yaml (called once at import time)."""
    import yaml

    for p in [
        pathlib.Path("config/config.yaml"),
        pathlib.Path("../config/config.yaml"),
        pathlib.Path("../../config/config.yaml"),
    ]:
        if p.exists():
            cfg = yaml.safe_load(p.read_text()) or {}
            db_section = cfg.get("database", {})
            if db_section:
                # Only override keys not already set via env vars
                if not os.environ.get("PGHOST"):
                    _db_config["host"] = str(db_section.get("host", _db_config["host"]))
                if not os.environ.get("PGPORT"):
                    _db_config["port"] = str(db_section.get("port", _db_config["port"]))
                if not os.environ.get("PGUSER"):
                    _db_config["user"] = str(db_section.get("user", _db_config["user"]))
                if not os.environ.get("PGPASSWORD"):
                    _db_config["password"] = str(db_section.get("password", _db_config["password"]))
                if not os.environ.get("PGDATABASE"):
                    _db_config["dbname"] = str(db_section.get("name", _db_config["dbname"]))
            return


# Auto-load from config.yaml at import time
_load_db_config_from_yaml()


def get_connection_string(dbname: str | None = None) -> str:
    """Get database connection string, optionally overriding the database name."""
    # If DATABASE_URL is explicitly set, honour it (legacy compat)
    env_url = os.environ.get("DATABASE_URL")
    if env_url and dbname is None:
        return env_url

    cfg = _db_config.copy()
    if dbname:
        cfg["dbname"] = dbname
    user = cfg["user"]
    password = cfg["password"]
    host = cfg["host"]
    port = cfg["port"]
    dbname = cfg["dbname"]
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def get_active_database() -> str:
    """Return the name of the currently active database."""
    return _db_config["dbname"]


def switch_database(dbname: str) -> str:
    """Switch the active database. All subsequent connections will use *dbname*.

    Returns the previous database name so callers can restore it.
    """
    previous = _db_config["dbname"]
    _db_config["dbname"] = dbname
    return previous


@contextmanager
def get_db() -> Generator[psycopg2.extensions.connection, None, None]:
    """Get a database connection with pgvector support."""
    conn = psycopg2.connect(get_connection_string())
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()


def get_admin_connection(dbname: str = "postgres"):
    """Get a connection to the admin (postgres) database for DDL operations."""
    conn = psycopg2.connect(get_connection_string(dbname=dbname))
    conn.autocommit = True  # Required for CREATE/DROP DATABASE
    return conn


# =============================================================================
# Papers CRUD
# =============================================================================

def _ensure_structured_summary_column(conn: psycopg2.extensions.connection) -> None:
    """Ensure structured_summary column exists on papers table."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE papers ADD COLUMN IF NOT EXISTS structured_summary JSONB"
        )
        conn.commit()


def create_paper(
    arxiv_id: str,
    title: str,
    authors: list[str],
    abstract: Optional[str] = None,
    summary: Optional[str] = None,
    abbreviation: Optional[str] = None,
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
                INSERT INTO papers (arxiv_id, title, authors, abstract, summary, abbreviation,
                                    pdf_path, latex_path, pdf_url, published_at, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    authors = EXCLUDED.authors,
                    abstract = EXCLUDED.abstract,
                    summary = COALESCE(EXCLUDED.summary, papers.summary),
                    abbreviation = COALESCE(EXCLUDED.abbreviation, papers.abbreviation),
                    pdf_path = COALESCE(EXCLUDED.pdf_path, papers.pdf_path),
                    latex_path = COALESCE(EXCLUDED.latex_path, papers.latex_path),
                    pdf_url = COALESCE(EXCLUDED.pdf_url, papers.pdf_url),
                    published_at = COALESCE(EXCLUDED.published_at, papers.published_at),
                    embedding = COALESCE(EXCLUDED.embedding, papers.embedding)
                RETURNING id
                """,
                (arxiv_id, title, authors, abstract, summary, abbreviation, pdf_path, 
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


def get_existing_arxiv_ids(arxiv_ids: list[str]) -> set[str]:
    """Check which arXiv IDs already exist in the database. Returns set of existing IDs."""
    if not arxiv_ids:
        return set()
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT arxiv_id FROM papers WHERE arxiv_id = ANY(%s)",
                (arxiv_ids,)
            )
            return {row[0] for row in cur.fetchall()}


def get_paper_by_id(paper_id: int) -> Optional[dict[str, Any]]:
    """Get paper by ID."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM papers WHERE id = %s", (paper_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_all_papers() -> list[dict[str, Any]]:
    """Get all papers."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM papers ORDER BY created_at DESC")
            return [dict(row) for row in cur.fetchall()]


def get_papers_missing_embeddings() -> list[str]:
    """Get arxiv_ids of papers that have NULL embeddings.

    Much faster than get_all_papers() because it avoids loading
    the huge 4096-float embedding vectors for all papers.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT arxiv_id FROM papers WHERE embedding IS NULL")
            return [row[0] for row in cur.fetchall()]


def get_papers_lightweight(columns: list[str] | None = None) -> dict[int, dict[str, Any]]:
    """Bulk-load lightweight paper data (no embedding) indexed by paper ID.

    This avoids the N+1 query problem in tree naming where individual
    get_paper_by_id() calls were creating 4000+ DB connections.

    Args:
        columns: Specific columns to select. Defaults to id, title, summary,
                 abbreviation, arxiv_id (excludes the 4096-float embedding).

    Returns:
        Dict mapping paper_id -> paper data dict.
    """
    if columns is None:
        columns = ["id", "title", "summary", "abbreviation", "arxiv_id"]
    # Always include 'id' for indexing
    if "id" not in columns:
        columns = ["id"] + columns
    cols_sql = ", ".join(columns)
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT {cols_sql} FROM papers")
            rows = cur.fetchall()
            return {int(row["id"]): dict(row) for row in rows}


def update_paper_embedding(paper_id: int, embedding: list[float]) -> None:
    """Update paper embedding."""
    import numpy as np
    embedding_arr = np.array(embedding, dtype=np.float32)
    
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE papers SET embedding = %s::vector WHERE id = %s",
                (embedding_arr, paper_id)
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


def update_paper_abbreviation(paper_id: int, abbreviation: str) -> None:
    """Update paper abbreviation (short display name)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE papers SET abbreviation = %s WHERE id = %s",
                (abbreviation, paper_id)
            )
            conn.commit()


def get_paper_abbreviation(paper_id: int) -> Optional[str]:
    """Get paper abbreviation by paper ID."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT abbreviation FROM papers WHERE id = %s", (paper_id,))
            row = cur.fetchone()
            return row[0] if row and row[0] else None


def update_paper_structured_summary(paper_id: int, structured_summary: dict) -> None:
    """Update paper structured summary (detailed analysis)."""
    import json
    with get_db() as conn:
        _ensure_structured_summary_column(conn)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE papers SET structured_summary = %s WHERE id = %s",
                (json.dumps(structured_summary), paper_id)
            )
            conn.commit()


def get_paper_structured_summary(paper_id: int) -> Optional[dict]:
    """Get paper structured summary."""
    import json
    with get_db() as conn:
        _ensure_structured_summary_column(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT structured_summary FROM papers WHERE id = %s",
                (paper_id,)
            )
            row = cur.fetchone()
            if row and row[0]:
                return row[0] if isinstance(row[0], dict) else json.loads(row[0])
            return None


def find_similar_papers(embedding: list[float], limit: int = 5, exclude_id: Optional[int] = None) -> list[dict[str, Any]]:
    """Find similar papers by embedding using cosine distance."""
    import numpy as np
    # Convert to numpy array for pgvector compatibility
    embedding_arr = np.array(embedding, dtype=np.float32)
    
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if exclude_id:
                cur.execute(
                    """
                    SELECT id, arxiv_id, title, authors, abstract, summary,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM papers
                    WHERE embedding IS NOT NULL AND id != %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding_arr, exclude_id, embedding_arr, limit)
                )
            else:
                cur.execute(
                    """
                    SELECT id, arxiv_id, title, authors, abstract, summary,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM papers
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding_arr, embedding_arr, limit)
                )
            return [dict(row) for row in cur.fetchall()]



# =============================================================================
# Paper Chunks CRUD (for custom RAG)
# =============================================================================

def store_paper_chunks(paper_id: int, chunks: list[dict]) -> None:
    """Bulk-insert text chunks with embeddings for a paper.

    Each chunk dict must have: text, chunk_index, char_start, char_end.
    Optional: embedding (list[float]).
    Existing chunks for the paper are deleted first.
    """
    import numpy as np

    with get_db() as conn:
        with conn.cursor() as cur:
            # Remove old chunks (idempotent re-index)
            cur.execute("DELETE FROM paper_chunks WHERE paper_id = %s", (paper_id,))

            for chunk in chunks:
                emb = chunk.get("embedding")
                emb_arr = np.array(emb, dtype=np.float32) if emb is not None else None
                cur.execute(
                    """
                    INSERT INTO paper_chunks
                        (paper_id, chunk_index, text, embedding, char_start, char_end)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        paper_id,
                        chunk["chunk_index"],
                        chunk["text"],
                        emb_arr,
                        chunk.get("char_start"),
                        chunk.get("char_end"),
                    ),
                )
            conn.commit()


def has_paper_chunks(paper_id: int) -> bool:
    """Return True if the paper has indexed chunks in paper_chunks."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM paper_chunks WHERE paper_id = %s)",
                (paper_id,),
            )
            return cur.fetchone()[0]


def get_paper_chunks(paper_id: int) -> list[dict]:
    """Return all chunks for a paper, ordered by chunk_index."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, paper_id, chunk_index, text, embedding,
                       char_start, char_end, created_at
                FROM paper_chunks
                WHERE paper_id = %s
                ORDER BY chunk_index
                """,
                (paper_id,),
            )
            return [dict(row) for row in cur.fetchall()]


def search_paper_chunks(
    paper_id: int,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[dict]:
    """Return top-k most similar chunks for a paper via cosine distance.

    Returns dicts with keys: id, text, chunk_index, similarity.
    """
    import numpy as np

    q_arr = np.array(query_embedding, dtype=np.float32)
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, chunk_index, text, char_start, char_end,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM paper_chunks
                WHERE paper_id = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (q_arr, paper_id, q_arr, top_k),
            )
            return [dict(row) for row in cur.fetchall()]


def delete_paper_chunks(paper_id: int) -> int:
    """Delete all chunks for a paper. Returns number of rows deleted."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM paper_chunks WHERE paper_id = %s", (paper_id,)
            )
            deleted = cur.rowcount
            conn.commit()
            return deleted


# =============================================================================
# Tree CRUD
# =============================================================================

def get_tree() -> dict[str, Any]:
    """Get full tree structure from JSONB and enrich with paper metadata.
    
    OPTIMIZED: Fetches all papers in a single query instead of N+1 queries.
    
    Returns:
        Frontend format tree structure with paper metadata:
        {
            "name": "AI Papers",
            "children": [
                {
                    "name": "Category Name",
                    "node_id": "node_abc",
                    "node_type": "category",
                    "children": [...]
                },
                {
                    "name": "Paper Name",
                    "node_id": "node_def",
                    "node_type": "paper",
                    "paper_id": 123,
                    "attributes": {
                        "arxivId": "...",
                        "title": "...",
                        ...
                    }
                }
            ]
        }
    """
    import json
    
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Fetch tree structure
            cur.execute("SELECT tree_data, node_names FROM tree_state WHERE id = 1")
            row = cur.fetchone()
            if not row or not row.get("tree_data"):
                return {"name": "AI Papers", "children": []}
            
            tree_data = row["tree_data"]
            tree_structure = tree_data if isinstance(tree_data, dict) else json.loads(tree_data)
            node_names_raw = row.get("node_names") or {}
            node_names = node_names_raw if isinstance(node_names_raw, dict) else json.loads(node_names_raw)
            
            # OPTIMIZATION: Fetch ALL papers in a single query and build lookup map
            cur.execute(
                """SELECT id, arxiv_id, title, authors, summary, pdf_path, abbreviation 
                   FROM papers"""
            )
            all_papers = cur.fetchall()
            papers_by_id = {p["id"]: p for p in all_papers}
    
    # Enrich tree with paper metadata using pre-fetched lookup
    def enrich_node(node: dict[str, Any]) -> dict[str, Any]:
        """Recursively enrich nodes with paper metadata."""
        if node.get("node_type") == "paper" and node.get("paper_id"):
            # Use pre-fetched paper data (O(1) lookup instead of DB query)
            paper = papers_by_id.get(node["paper_id"])
            if paper:
                node["attributes"] = {
                    "arxivId": paper.get("arxiv_id"),
                    "title": paper.get("title"),
                    "authors": paper.get("authors") or [],
                    "summary": paper.get("summary"),
                    "pdfPath": paper.get("pdf_path"),
                }
                # Use stored abbreviation from database (preferred)
                db_abbrev = paper.get("abbreviation")
                if db_abbrev:
                    node["name"] = db_abbrev
                else:
                    # Fallback: use existing node name or generate from title
                    existing_name = (node.get("name") or "").strip()
                    if not existing_name or existing_name.startswith("paper_"):
                        title = paper.get("title", "")
                        if title:
                            # Extract abbreviation if present before colon
                            if ":" in title:
                                prefix = title.split(":")[0].strip()
                                if len(prefix) <= 20:
                                    node["name"] = prefix
                                else:
                                    node["name"] = title[:20] + ".."
                            else:
                                node["name"] = title[:20] + ".." if len(title) > 20 else title
        elif node.get("node_type") == "category":
            node_id = node.get("node_id")
            if node_id and node_names:
                named = (node_names.get(node_id) or "").strip()
                if named:
                    current = (node.get("name") or "").strip()
                    if not current or current == node_id or current.startswith("node_"):
                        node["name"] = named
        
        # Recursively enrich children
        if node.get("children"):
            node["children"] = [enrich_node(child) for child in node["children"]]
        
        return node
    
    return enrich_node(tree_structure)


def save_tree(tree_structure: dict[str, Any], node_names: dict[str, str] | None = None) -> None:
    """Save tree structure to database as JSONB.
    
    Args:
        tree_structure: Frontend format tree structure {name, children: [...]}
        node_names: Optional dictionary mapping node_id to node name
    """
    import json
    
    if node_names is None:
        node_names = {}
        def extract_names(node: dict[str, Any]):
            if node.get("node_id"):
                node_names[node["node_id"]] = node.get("name", "")
            if node.get("children"):
                for child in node["children"]:
                    extract_names(child)
        extract_names(tree_structure)
    
    with get_db() as conn:
        with conn.cursor() as cur:
            # Lock the row to prevent concurrent updates
            cur.execute("SELECT * FROM tree_state WHERE id = 1 FOR UPDATE")
            
            cur.execute(
                """
                UPDATE tree_state 
                SET tree_data = %s::jsonb, node_names = %s::jsonb, updated_at = NOW()
                WHERE id = 1
                """,
                (json.dumps(tree_structure), json.dumps(node_names))
            )
            conn.commit()


def get_tree_node_names() -> dict[str, str]:
    """Get node names mapping from database.
    
    Returns:
        Dictionary mapping node_id -> name
    """
    import json
    
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT node_names FROM tree_state WHERE id = 1")
            row = cur.fetchone()
            if not row or not row.get("node_names"):
                return {}
            
            node_names_data = row["node_names"]
            return node_names_data if isinstance(node_names_data, dict) else json.loads(node_names_data)


def find_paper_node_id(paper_id: int) -> str | None:
    """Find the node_id for a paper in the tree.
    
    Args:
        paper_id: Integer paper ID
        
    Returns:
        Node ID if found, None otherwise
    """
    tree = get_tree()
    
    def search_node(node: dict[str, Any]) -> str | None:
        if node.get("node_type") == "paper" and node.get("paper_id") == paper_id:
            return node.get("node_id")
        if node.get("children"):
            for child in node["children"]:
                result = search_node(child)
                if result:
                    return result
        return None
    
    return search_node(tree)


def update_tree_node_name(node_id: str, name: str) -> None:
    """Update a tree node's display name in JSONB structure.
    
    Args:
        node_id: Node ID to update
        name: New name for the node
    """
    import json
    
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Lock the row to prevent concurrent updates
            cur.execute("SELECT tree_data, node_names FROM tree_state WHERE id = 1 FOR UPDATE")
            row = cur.fetchone()
            if not row:
                return
            
            tree_data_raw = row["tree_data"]
            node_names_raw = row["node_names"]
            tree_data = tree_data_raw if isinstance(tree_data_raw, dict) else json.loads(tree_data_raw)
            node_names = node_names_raw if isinstance(node_names_raw, dict) else json.loads(node_names_raw)
            
            # Update name in tree structure
            def update_name_in_tree(node: dict[str, Any]) -> None:
                if node.get("node_id") == node_id:
                    node["name"] = name
                if node.get("children"):
                    for child in node["children"]:
                        update_name_in_tree(child)
            
            update_name_in_tree(tree_data)
            
            # Update node_names mapping
            node_names[node_id] = name
            
            # Save back
            cur.execute(
                """
                UPDATE tree_state 
                SET tree_data = %s::jsonb, node_names = %s::jsonb, updated_at = NOW()
                WHERE id = 1
                """,
                (json.dumps(tree_data), json.dumps(node_names))
            )
            conn.commit()


def delete_paper(paper_id: int) -> None:
    """Delete a paper and all its associated data.
    
    CASCADE constraints will automatically delete:
    - paper_references (source_paper_id)
    - similar_papers_cache (paper_id)
    - repo_cache (paper_id)
    - paper_queries (paper_id)
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM papers WHERE id = %s", (paper_id,))
            conn.commit()




def get_all_categories_with_counts() -> list[dict[str, Any]]:
    """Get all categories with their paper counts from tree structure."""
    tree = get_tree()
    categories = []
    
    def traverse(node: dict[str, Any]) -> None:
        if node.get("node_type") == "category":
            # Count papers in this category
            paper_count = 0
            def count_papers(n: dict[str, Any]) -> int:
                count = 0
                if n.get("node_type") == "paper":
                    return 1
                if n.get("children"):
                    for child in n["children"]:
                        count += count_papers(child)
                return count
            
            paper_count = count_papers(node)
            categories.append({
                "node_id": node.get("node_id"),
                "name": node.get("name"),
                "paper_count": paper_count,
            })
        
        if node.get("children"):
            for child in node["children"]:
                traverse(child)
    
    traverse(tree)
    return sorted(categories, key=lambda x: x["paper_count"], reverse=True)


def get_category_paper_count(category_name: str) -> int:
    """Get number of papers in a category by name."""
    tree = get_tree()
    
    def find_category(node: dict[str, Any]) -> dict[str, Any] | None:
        if node.get("node_type") == "category" and node.get("name") == category_name:
            return node
        if node.get("children"):
            for child in node["children"]:
                result = find_category(child)
                if result:
                    return result
        return None
    
    category = find_category(tree)
    if not category:
        return 0
    
    def count_papers(n: dict[str, Any]) -> int:
        count = 0
        if n.get("node_type") == "paper":
            return 1
        if n.get("children"):
            for child in n["children"]:
                count += count_papers(child)
        return count
    
    return count_papers(category)


def get_category_paper_count_by_id(category_node_id: str) -> int:
    """Get number of papers in a category by node_id."""
    tree = get_tree()
    
    def find_category(node: dict[str, Any]) -> dict[str, Any] | None:
        if node.get("node_id") == category_node_id:
            return node
        if node.get("children"):
            for child in node["children"]:
                result = find_category(child)
                if result:
                    return result
        return None
    
    category = find_category(tree)
    if not category:
        return 0
    
    def count_papers(n: dict[str, Any]) -> int:
        count = 0
        if n.get("node_type") == "paper":
            return 1
        if n.get("children"):
            for child in n["children"]:
                count += count_papers(child)
        return count
    
    return count_papers(category)


def get_papers_in_category(category_node_id: str) -> list[dict[str, Any]]:
    """Get all papers in a category with their details."""
    tree = get_tree()
    papers = []
    
    def find_category(node: dict[str, Any]) -> dict[str, Any] | None:
        if node.get("node_id") == category_node_id:
            return node
        if node.get("children"):
            for child in node["children"]:
                result = find_category(child)
                if result:
                    return result
        return None
    
    category = find_category(tree)
    if not category:
        return []
    
    def collect_papers(n: dict[str, Any]) -> None:
        if n.get("node_type") == "paper" and n.get("paper_id"):
            paper = get_paper_by_id(n["paper_id"])
            if paper:
                papers.append({
                    "node_id": n.get("node_id"),
                    "display_name": n.get("name"),
                    **paper,
                })
        if n.get("children"):
            for child in n["children"]:
                collect_papers(child)
    
    collect_papers(category)
    return papers



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


def get_reference_by_id(ref_id: int) -> Optional[dict[str, Any]]:
    """Get a single reference row by its primary key."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM paper_references WHERE id = %s",
                (ref_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None


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
    """Get cached similar papers with field names matching frontend expectations."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT 
                    similar_arxiv_id as arxiv_id,
                    similar_title as title,
                    similarity_score as similarity,
                    description
                FROM similar_papers_cache 
                WHERE paper_id = %s 
                ORDER BY similarity_score DESC NULLS LAST
                """,
                (paper_id,)
            )
            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Query History
# =============================================================================

def add_query(
    paper_id: int,
    question: str,
    answer: str,
    model: Optional[str] = None,
) -> int:
    """Save a QA query and return its ID."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO paper_queries (paper_id, question, answer, model)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (paper_id, question, answer, model)
            )
            query_id = cur.fetchone()[0]
            conn.commit()
            return query_id


def get_queries(paper_id: int) -> list[dict[str, Any]]:
    """Get query history for a paper."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM paper_queries WHERE paper_id = %s ORDER BY created_at DESC",
                (paper_id,)
            )
            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Get All Cached Data for a Paper
# =============================================================================

def get_paper_cached_data(paper_id: int) -> dict[str, Any]:
    """Get all cached data for a paper (repos, refs, similar, queries)."""
    return {
        "repos": get_cached_repos(paper_id),
        "references": get_references(paper_id),
        "similar_papers": get_cached_similar_papers(paper_id),
        "queries": get_queries(paper_id),
    }


# =============================================================================
# Topic Query CRUD
# =============================================================================

def create_topic(
    name: str,
    topic_query: str,
    embedding: Optional[list[float]] = None,
) -> int:
    """Create a topic and return its ID."""
    import numpy as np
    
    with get_db() as conn:
        with conn.cursor() as cur:
            if embedding:
                embedding_arr = np.array(embedding, dtype=np.float32)
                cur.execute(
                    """
                    INSERT INTO topics (name, topic_query, embedding)
                    VALUES (%s, %s, %s::vector)
                    RETURNING id
                    """,
                    (name, topic_query, embedding_arr)
                )
            else:
                cur.execute(
                    """
                    INSERT INTO topics (name, topic_query)
                    VALUES (%s, %s)
                    RETURNING id
                    """,
                    (name, topic_query)
                )
            topic_id = cur.fetchone()[0]
            conn.commit()
            return topic_id


def get_topic_by_id(topic_id: int) -> Optional[dict[str, Any]]:
    """Get topic by ID with paper and query counts.

    Note: excludes the embedding column to avoid numpy.ndarray
    serialization errors in Pydantic/FastAPI responses.
    """
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT t.id, t.name, t.topic_query, t.created_at, t.updated_at,
                       (SELECT COUNT(*) FROM topic_papers tp WHERE tp.topic_id = t.id) as paper_count,
                       (SELECT COUNT(*) FROM topic_queries tq WHERE tq.topic_id = t.id) as query_count
                FROM topics t
                WHERE t.id = %s
                """,
                (topic_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None


def get_topic_by_name(name: str) -> Optional[dict[str, Any]]:
    """Get topic by name.

    Note: excludes the embedding column to avoid numpy.ndarray
    serialization errors in Pydantic/FastAPI responses.
    """
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, name, topic_query, created_at, updated_at FROM topics WHERE name = %s",
                (name,)
            )
            row = cur.fetchone()
            return dict(row) if row else None


def get_topics_by_query(topic_query: str) -> list[dict[str, Any]]:
    """Get all topics matching a topic query string.

    Note: excludes the embedding column to avoid numpy.ndarray
    serialization errors in Pydantic/FastAPI responses.
    """
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT t.id, t.name, t.topic_query, t.created_at, t.updated_at,
                       (SELECT COUNT(*) FROM topic_papers tp WHERE tp.topic_id = t.id) as paper_count,
                       (SELECT COUNT(*) FROM topic_queries tq WHERE tq.topic_id = t.id) as query_count
                FROM topics t
                WHERE t.topic_query = %s
                ORDER BY t.created_at DESC
                """,
                (topic_query,)
            )
            return [dict(row) for row in cur.fetchall()]


def get_all_topics() -> list[dict[str, Any]]:
    """Get all topics ordered by recency with paper and query counts."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT t.id, t.name, t.topic_query, t.created_at, t.updated_at,
                       (SELECT COUNT(*) FROM topic_papers tp WHERE tp.topic_id = t.id) as paper_count,
                       (SELECT COUNT(*) FROM topic_queries tq WHERE tq.topic_id = t.id) as query_count
                FROM topics t
                ORDER BY t.created_at DESC
                """
            )
            return [dict(row) for row in cur.fetchall()]


def delete_topic(topic_id: int) -> bool:
    """Delete a topic (cascades to topic_papers and topic_queries)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM topics WHERE id = %s RETURNING id", (topic_id,))
            deleted = cur.fetchone() is not None
            conn.commit()
            return deleted


def add_papers_to_topic(topic_id: int, paper_ids: list[int], similarity_scores: Optional[list[float]] = None) -> int:
    """Add papers to a topic pool. Returns number of papers added."""
    added = 0
    with get_db() as conn:
        with conn.cursor() as cur:
            for i, paper_id in enumerate(paper_ids):
                similarity = similarity_scores[i] if similarity_scores and i < len(similarity_scores) else None
                try:
                    cur.execute(
                        """
                        INSERT INTO topic_papers (topic_id, paper_id, similarity_score)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (topic_id, paper_id) DO NOTHING
                        """,
                        (topic_id, paper_id, similarity)
                    )
                    if cur.rowcount > 0:
                        added += 1
                except Exception:
                    pass  # Skip duplicates
            conn.commit()
    return added


def remove_paper_from_topic(topic_id: int, paper_id: int) -> bool:
    """Remove a paper from a topic pool."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM topic_papers WHERE topic_id = %s AND paper_id = %s RETURNING id",
                (topic_id, paper_id)
            )
            deleted = cur.fetchone() is not None
            conn.commit()
            return deleted


def get_topic_papers(topic_id: int) -> list[dict[str, Any]]:
    """Get all papers in a topic pool with paper details."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT p.id as paper_id, p.arxiv_id, p.title, p.authors, p.summary, p.pdf_path,
                       tp.similarity_score, tp.added_at,
                       EXTRACT(YEAR FROM p.published_at) as year
                FROM topic_papers tp
                JOIN papers p ON p.id = tp.paper_id
                WHERE tp.topic_id = %s
                ORDER BY tp.similarity_score DESC NULLS LAST
                """,
                (topic_id,)
            )
            return [dict(row) for row in cur.fetchall()]


def add_topic_query(
    topic_id: int,
    question: str,
    answer: str,
    paper_responses: Optional[dict] = None,
    model: Optional[str] = None,
) -> int:
    """Save a topic query and return its ID."""
    import json
    
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO topic_queries (topic_id, question, answer, paper_responses, model)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (topic_id, question, answer, json.dumps(paper_responses) if paper_responses else None, model)
            )
            query_id = cur.fetchone()[0]
            conn.commit()
            return query_id


def get_topic_queries(topic_id: int) -> list[dict[str, Any]]:
    """Get query history for a topic."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM topic_queries WHERE topic_id = %s ORDER BY created_at DESC",
                (topic_id,)
            )
            return [dict(row) for row in cur.fetchall()]


def get_topic_full(topic_id: int) -> Optional[dict[str, Any]]:
    """Get full topic details with papers and queries."""
    topic = get_topic_by_id(topic_id)
    if not topic:
        return None
    
    topic["papers"] = get_topic_papers(topic_id)
    topic["queries"] = get_topic_queries(topic_id)
    return topic


def search_papers_by_embedding(
    embedding: list[float],
    limit: int = 10,
    offset: int = 0,
    min_similarity: float = 0.0,
    exclude_paper_ids: Optional[list[int]] = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Search papers by embedding similarity.

    If min_similarity > 0 and the threshold filters out ALL papers,
    automatically falls back to returning the top-K results without
    threshold filtering. This prevents empty results for short or
    specific queries (e.g. "FP8") whose embeddings produce lower
    cosine similarity scores.
    
    Returns:
        Tuple of (papers list, has_more bool)
    """
    import numpy as np
    embedding_arr = np.array(embedding, dtype=np.float32)
    
    def _run_query(conn, threshold: float) -> list[dict[str, Any]]:
        """Run the similarity search with given threshold."""
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Build exclusion clause
            exclusion_clause = ""
            base_params: list = [embedding_arr, embedding_arr, threshold]
            
            if exclude_paper_ids:
                placeholders = ",".join(["%s"] * len(exclude_paper_ids))
                exclusion_clause = f"AND p.id NOT IN ({placeholders})"
                base_params.extend(exclude_paper_ids)
            
            base_params.extend([limit + 1, offset])
            
            # Build final params: embed, embed, threshold, [exclusions], embed, limit, offset
            final_params = base_params[:3] + (base_params[3:-2] if exclude_paper_ids else []) + [embedding_arr] + base_params[-2:]
            
            cur.execute(
                f"""
                SELECT p.id as paper_id, p.arxiv_id, p.title, 
                       EXTRACT(YEAR FROM p.published_at) as year,
                       1 - (p.embedding <=> %s::vector) AS similarity
                FROM papers p
                WHERE p.embedding IS NOT NULL
                  AND 1 - (p.embedding <=> %s::vector) >= %s
                  {exclusion_clause}
                ORDER BY p.embedding <=> %s::vector
                LIMIT %s OFFSET %s
                """,
                final_params,
            )
            return [dict(row) for row in cur.fetchall()]
    
    with get_db() as conn:
        results = _run_query(conn, min_similarity)
        
        # Fallback: if threshold filtered out everything, retry without threshold
        if not results and min_similarity > 0 and offset == 0:
            results = _run_query(conn, 0.0)
        
        has_more = len(results) > limit
        if has_more:
            results = results[:limit]
        
        return results, has_more


# =============================================================================
# Settings CRUD Functions
# =============================================================================

def get_all_settings() -> dict[str, Any]:
    """Get all settings from database as a dict keyed by setting key."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT key, value, category, updated_at FROM settings")
            rows = cur.fetchall()
            return {row["key"]: {"value": row["value"], "category": row["category"], "updated_at": row["updated_at"]} for row in rows}


def get_setting(key: str) -> Optional[str]:
    """Get a single setting value by key."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT value FROM settings WHERE key = %s", (key,))
            row = cur.fetchone()
            return row["value"] if row else None


def set_setting(key: str, value: str, category: str, updated_by: Optional[str] = None) -> None:
    """Set a setting value (upsert)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO settings (key, value, category, updated_by)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    category = EXCLUDED.category,
                    updated_by = EXCLUDED.updated_by,
                    updated_at = NOW()
                """,
                (key, value, category, updated_by)
            )
        conn.commit()


def set_settings_batch(settings: list[dict[str, str]], updated_by: Optional[str] = None) -> None:
    """Set multiple settings at once."""
    with get_db() as conn:
        with conn.cursor() as cur:
            for s in settings:
                cur.execute(
                    """
                    INSERT INTO settings (key, value, category, updated_by)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        category = EXCLUDED.category,
                        updated_by = EXCLUDED.updated_by,
                        updated_at = NOW()
                    """,
                    (s["key"], s["value"], s["category"], updated_by)
                )
        conn.commit()


def delete_setting(key: str) -> bool:
    """Delete a single setting."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM settings WHERE key = %s", (key,))
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted


def clear_all_settings() -> int:
    """Clear all settings (reset to defaults from config.yaml)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM settings")
            deleted = cur.rowcount
        conn.commit()
        return deleted
