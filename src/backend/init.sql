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
    structured_summary JSONB,  -- Structured analysis results (cached)
    abbreviation VARCHAR(50),  -- Short name for display in tree (e.g., "mHC", "GPT-4")
    pdf_path TEXT,
    latex_path TEXT,
    pdf_url TEXT,
    published_at TIMESTAMPTZ,
    embedding vector(4096),  -- Qwen3-Embedding dimension
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tree state table: stores the taxonomy tree structure as JSONB
CREATE TABLE IF NOT EXISTS tree_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    tree_data JSONB NOT NULL DEFAULT '{"name": "AI Papers", "children": []}'::jsonb,
    node_names JSONB NOT NULL DEFAULT '{}'::jsonb,  -- Mapping of node_id -> name
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT single_tree CHECK (id = 1)  -- Only allow one row
);

-- Create index for JSONB queries
CREATE INDEX IF NOT EXISTS idx_tree_state_gin ON tree_state USING GIN (tree_data);

-- Insert initial empty tree
INSERT INTO tree_state (id, tree_data, node_names)
VALUES (1, '{"name": "AI Papers", "children": []}'::jsonb, '{}'::jsonb)
ON CONFLICT (id) DO NOTHING;

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

-- QA queries cache: stores question-answer history for papers
CREATE TABLE IF NOT EXISTS paper_queries (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    model VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_queries_paper ON paper_queries(paper_id);


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

DROP TRIGGER IF EXISTS update_tree_state_updated_at ON tree_state;
CREATE TRIGGER update_tree_state_updated_at
    BEFORE UPDATE ON tree_state
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Topic Query Tables: Multi-paper RAG queries by topic
-- =============================================================================

-- Topics table: paper pools organized by topic
CREATE TABLE IF NOT EXISTS topics (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,           -- User-provided prefix + topic (unique identifier)
    topic_query TEXT NOT NULL,            -- Original search topic
    embedding vector(4096),               -- Topic embedding for similarity search
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name)
);

CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name);

-- Topic papers: papers in each topic pool
CREATE TABLE IF NOT EXISTS topic_papers (
    id SERIAL PRIMARY KEY,
    topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    similarity_score FLOAT,               -- Similarity to topic when added
    added_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(topic_id, paper_id)
);

CREATE INDEX IF NOT EXISTS idx_topic_papers_topic ON topic_papers(topic_id);
CREATE INDEX IF NOT EXISTS idx_topic_papers_paper ON topic_papers(paper_id);

-- Topic queries: Q&A history for each topic
CREATE TABLE IF NOT EXISTS topic_queries (
    id SERIAL PRIMARY KEY,
    topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    paper_responses JSONB,                -- Individual paper responses for debugging
    model VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_topic_queries_topic ON topic_queries(topic_id);

-- Trigger for topics updated_at
DROP TRIGGER IF EXISTS update_topics_updated_at ON topics;
CREATE TRIGGER update_topics_updated_at
    BEFORE UPDATE ON topics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Global Settings Table: Runtime configuration overrides
-- =============================================================================

-- Settings table: stores runtime config overrides (DB takes precedence over config.yaml)
CREATE TABLE IF NOT EXISTS settings (
    id SERIAL PRIMARY KEY,
    key VARCHAR(100) UNIQUE NOT NULL,         -- Setting key (e.g., 'llm_base_url', 'skip_existing')
    value TEXT NOT NULL,                       -- Setting value (stored as text, parsed by application)
    category VARCHAR(50) NOT NULL,             -- Category for UI grouping (e.g., 'llm', 'ingestion', 'paperqa')
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by VARCHAR(100)                    -- Optional: who made the change
);

CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key);
CREATE INDEX IF NOT EXISTS idx_settings_category ON settings(category);

-- Trigger for settings updated_at
DROP TRIGGER IF EXISTS update_settings_updated_at ON settings;
CREATE TRIGGER update_settings_updated_at
    BEFORE UPDATE ON settings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Paper Chunks Table: chunk-level text + embeddings for RAG
-- =============================================================================

CREATE TABLE IF NOT EXISTS paper_chunks (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,       -- ordering within the paper
    text TEXT NOT NULL,                  -- the chunk text
    embedding vector(4096),             -- chunk-level embedding
    char_start INTEGER,                 -- position in original text
    char_end INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(paper_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_paper_chunks_paper ON paper_chunks(paper_id);

-- =============================================================================
-- Incremental Placement Tables: dirty tracking + category centroids
-- =============================================================================

-- Dirty tree nodes: nodes that received new papers via placement since last recategorize
CREATE TABLE IF NOT EXISTS dirty_tree_nodes (
    node_id TEXT PRIMARY KEY,
    added_at TIMESTAMPTZ DEFAULT NOW()
);

-- Category embeddings: centroid embeddings per tree node for fast placement descent
CREATE TABLE IF NOT EXISTS category_embeddings (
    node_id TEXT PRIMARY KEY,
    embedding vector(4096) NOT NULL,
    paper_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Migrate settings category from 'classification' to 'categorization'
UPDATE settings SET category = 'categorization' WHERE category = 'classification';
