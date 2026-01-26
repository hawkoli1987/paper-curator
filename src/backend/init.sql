-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Papers table: stores all ingested papers
CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    arxiv_id VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT[] NOT NULL,
    abstract TEXT,
    summary TEXT,
    pdf_path TEXT,
    latex_path TEXT,
    pdf_url TEXT,
    published_at TIMESTAMPTZ,
    embedding vector(4096),  -- Qwen3-Embedding dimension
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tree nodes table: stores the taxonomy tree structure
CREATE TABLE IF NOT EXISTS tree_nodes (
    id SERIAL PRIMARY KEY,
    node_id VARCHAR(100) UNIQUE NOT NULL,  -- Unique identifier for the node
    name TEXT NOT NULL,
    node_type VARCHAR(20) NOT NULL CHECK (node_type IN ('root', 'category', 'paper')),
    parent_id VARCHAR(100),  -- References node_id of parent
    paper_id INTEGER REFERENCES papers(id) ON DELETE SET NULL,
    position INTEGER DEFAULT 0,  -- Order among siblings
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for parent lookups
CREATE INDEX IF NOT EXISTS idx_tree_nodes_parent ON tree_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_tree_nodes_type ON tree_nodes(node_type);

-- GitHub repos cache: stores repo lookup results
CREATE TABLE IF NOT EXISTS repo_cache (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    source VARCHAR(50) NOT NULL,  -- 'paperswithcode' or 'github'
    repo_url TEXT,
    repo_name TEXT,
    stars INTEGER,
    is_official BOOLEAN DEFAULT FALSE,
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_repo_cache_paper ON repo_cache(paper_id);

-- References table: stores extracted references from papers
CREATE TABLE IF NOT EXISTS paper_references (
    id SERIAL PRIMARY KEY,
    source_paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    cited_arxiv_id VARCHAR(50),  -- If we can resolve to arXiv
    cited_title TEXT NOT NULL,
    cited_authors TEXT[],
    cited_year INTEGER,
    citation_context TEXT,  -- The sentence/paragraph where it's cited
    explanation TEXT,  -- LLM-generated explanation (cached)
    explained_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_references_source ON paper_references(source_paper_id);
CREATE INDEX IF NOT EXISTS idx_references_arxiv ON paper_references(cited_arxiv_id);

-- Similar papers cache: stores similarity search results
CREATE TABLE IF NOT EXISTS similar_papers_cache (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    similar_arxiv_id VARCHAR(50),
    similar_title TEXT NOT NULL,
    similarity_score FLOAT,
    description TEXT,  -- Why it's similar
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_similar_paper ON similar_papers_cache(paper_id);

-- Insert root node for the tree
INSERT INTO tree_nodes (node_id, name, node_type, parent_id, position)
VALUES ('root', 'AI Papers', 'root', NULL, 0)
ON CONFLICT (node_id) DO NOTHING;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
DROP TRIGGER IF EXISTS update_papers_updated_at ON papers;
CREATE TRIGGER update_papers_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_tree_nodes_updated_at ON tree_nodes;
CREATE TRIGGER update_tree_nodes_updated_at
    BEFORE UPDATE ON tree_nodes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
