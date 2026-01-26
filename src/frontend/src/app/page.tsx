"use client";

import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import type { RawNodeDatum } from "react-d3-tree";

const Tree = dynamic(() => import("react-d3-tree"), { ssr: false });

interface PaperNode {
  name: string;
  node_id?: string;
  node_type?: string;
  children?: PaperNode[];
  attributes?: {
    arxivId?: string;
    title?: string;
    authors?: string[];
    summary?: string;
    pdfPath?: string;
    category?: string;
  };
}

interface IngestionStep {
  name: string;
  status: "pending" | "running" | "done" | "error";
  message?: string;
}

interface RepoResult {
  source: string;
  repo_url: string;
  repo_name: string;
  stars: number;
  is_official: boolean;
}

interface Reference {
  id: number;
  cited_title: string;
  cited_arxiv_id?: string;
  cited_authors?: string[];
  cited_year?: number;
  explanation?: string;
}

interface SimilarPaper {
  arxiv_id: string;
  title: string;
  similarity: number;
  authors?: string[];
}

interface ContextMenuState {
  visible: boolean;
  x: number;
  y: number;
  node: PaperNode | null;
}

interface UIConfig {
  hover_debounce_ms: number;
  max_similar_papers: number;
  tree_auto_save_interval_ms: number;
}

// Convert PaperNode to react-d3-tree compatible format
function toTreeData(node: PaperNode): RawNodeDatum {
  return {
    name: node.name,
    attributes: node.attributes
      ? {
          arxivId: node.attributes.arxivId || "",
          title: node.attributes.title || "",
          authors: node.attributes.authors?.join(", ") || "",
          category: node.attributes.category || "",
        }
      : undefined,
    children: node.children?.map(toTreeData),
  };
}

const initialTaxonomy: PaperNode = {
  name: "AI Papers",
  children: [],
};

export default function Home() {
  const [arxivUrl, setArxivUrl] = useState("");
  const [taxonomy, setTaxonomy] = useState<PaperNode>(initialTaxonomy);
  const [selectedNode, setSelectedNode] = useState<PaperNode | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [steps, setSteps] = useState<IngestionStep[]>([]);
  const [uiConfig, setUiConfig] = useState<UIConfig | null>(null);
  
  // Context menu state
  const [contextMenu, setContextMenu] = useState<ContextMenuState>({
    visible: false,
    x: 0,
    y: 0,
    node: null,
  });
  
  // Feature panel states
  const [activePanel, setActivePanel] = useState<"details" | "repos" | "references" | "similar">("details");
  const [repos, setRepos] = useState<RepoResult[]>([]);
  const [references, setReferences] = useState<Reference[]>([]);
  const [similarPapers, setSimilarPapers] = useState<SimilarPaper[]>([]);
  const [isLoadingFeature, setIsLoadingFeature] = useState(false);
  const [hoveredRefId, setHoveredRefId] = useState<number | null>(null);
  const [refExplanations, setRefExplanations] = useState<Record<number, string>>({});
  
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Load config and tree on mount
  useEffect(() => {
    const loadInitialData = async () => {
      // Load UI config
      try {
        const configRes = await fetch("/api/config");
        if (configRes.ok) {
          const config = await configRes.json();
          setUiConfig(config);
        }
      } catch (e) {
        console.error("Failed to load config:", e);
      }
      
      // Load tree from database
      try {
        const treeRes = await fetch("/api/tree");
        if (treeRes.ok) {
          const treeData = await treeRes.json();
          if (treeData.children && treeData.children.length > 0) {
            setTaxonomy(treeData);
          }
        }
      } catch (e) {
        console.error("Failed to load tree:", e);
      }
    };
    
    loadInitialData();
  }, []);

  // Close context menu on click outside
  useEffect(() => {
    const handleClick = () => setContextMenu((prev) => ({ ...prev, visible: false }));
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, []);

  const updateStep = (index: number, update: Partial<IngestionStep>) => {
    setSteps((prev) => prev.map((s, i) => (i === index ? { ...s, ...update } : s)));
  };

  // Get existing categories from the tree
  const existingCategories = useMemo(() => {
    return taxonomy.children?.map((c) => c.name) || [];
  }, [taxonomy]);

  const addPaperToTree = useCallback(
    (paper: {
      arxivId: string;
      title: string;
      authors: string[];
      summary: string;
      pdfPath?: string;
      category: string;
    }) => {
      setTaxonomy((prev) => {
        const newTree = JSON.parse(JSON.stringify(prev)) as PaperNode;
        let categoryNode = newTree.children?.find((c) => c.name === paper.category);
        if (!categoryNode) {
          categoryNode = { name: paper.category, children: [], node_type: "category" };
          newTree.children = newTree.children || [];
          newTree.children.push(categoryNode);
        }
        categoryNode.children = categoryNode.children || [];
        categoryNode.children.push({
          name: paper.title.length > 40 ? paper.title.slice(0, 40) + "..." : paper.title,
          node_type: "paper",
          attributes: {
            arxivId: paper.arxivId,
            title: paper.title,
            authors: paper.authors,
            summary: paper.summary,
            pdfPath: paper.pdfPath,
            category: paper.category,
          },
        });
        return newTree;
      });
    },
    []
  );

  const handleIngest = async () => {
    if (!arxivUrl.trim()) return;

    setIsIngesting(true);
    setSteps([
      { name: "Resolve arXiv metadata", status: "pending" },
      { name: "Download PDF", status: "pending" },
      { name: "Extract text", status: "pending" },
      { name: "Classify paper (LLM)", status: "pending" },
      { name: "Generate summary (LLM)", status: "pending" },
      { name: "Save to database", status: "pending" },
    ]);

    let arxivId = "";
    let title = "";
    let authors: string[] = [];
    let abstract = "";
    let pdfPath = "";
    let latexPath = "";
    let pdfUrl = "";
    let published = "";
    let category = "";
    let summary = "";

    // Step 1: Resolve
    updateStep(0, { status: "running" });
    const resolveRes = await fetch("/api/arxiv/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: arxivUrl }),
    });
    if (!resolveRes.ok) {
      updateStep(0, { status: "error", message: `HTTP ${resolveRes.status}` });
      setIsIngesting(false);
      return;
    }
    const resolveData = await resolveRes.json();
    arxivId = resolveData.arxiv_id;
    title = resolveData.title;
    authors = resolveData.authors;
    abstract = resolveData.summary;
    pdfUrl = resolveData.pdf_url;
    published = resolveData.published;
    updateStep(0, { status: "done", message: title });

    // Step 2: Download
    updateStep(1, { status: "running" });
    const downloadRes = await fetch("/api/arxiv/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arxiv_id: arxivId }),
    });
    if (!downloadRes.ok) {
      updateStep(1, { status: "error", message: `HTTP ${downloadRes.status}` });
      setIsIngesting(false);
      return;
    }
    const downloadData = await downloadRes.json();
    pdfPath = downloadData.pdf_path;
    latexPath = downloadData.latex_path;
    updateStep(1, { status: "done", message: "PDF downloaded" });

    // Step 3: Extract
    updateStep(2, { status: "running" });
    const extractRes = await fetch("/api/pdf/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pdf_path: pdfPath }),
    });
    if (!extractRes.ok) {
      updateStep(2, { status: "error", message: `HTTP ${extractRes.status}` });
      setIsIngesting(false);
      return;
    }
    updateStep(2, { status: "done", message: "Text extracted" });

    // Step 4: Classify (LLM)
    updateStep(3, { status: "running", message: "Determining category..." });
    const classifyRes = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title,
        abstract,
        existing_categories: existingCategories,
      }),
    });
    if (!classifyRes.ok) {
      updateStep(3, { status: "error", message: `HTTP ${classifyRes.status}` });
      setIsIngesting(false);
      return;
    }
    const classifyData = await classifyRes.json();
    category = classifyData.category;
    updateStep(3, { status: "done", message: `Category: ${category}` });

    // Step 5: Summarize
    updateStep(4, { status: "running", message: "Generating summary (this may take a minute)..." });
    const summarizeRes = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pdf_path: pdfPath }),
    });
    if (!summarizeRes.ok) {
      updateStep(4, { status: "error", message: `HTTP ${summarizeRes.status}` });
      setIsIngesting(false);
      return;
    }
    const summarizeData = await summarizeRes.json();
    summary = summarizeData.summary;
    updateStep(4, { status: "done", message: "Summary generated" });

    // Step 6: Save to database
    updateStep(5, { status: "running" });
    const saveRes = await fetch("/api/papers/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        arxiv_id: arxivId,
        title,
        authors,
        abstract,
        summary,
        pdf_path: pdfPath,
        latex_path: latexPath,
        pdf_url: pdfUrl,
        published_at: published,
        category,
      }),
    });
    if (!saveRes.ok) {
      updateStep(5, { status: "error", message: `HTTP ${saveRes.status}` });
      // Still add to local tree even if DB save fails
    } else {
      updateStep(5, { status: "done", message: "Saved" });
    }

    // Add to tree
    addPaperToTree({
      arxivId,
      title,
      authors,
      summary,
      pdfPath,
      category,
    });

    setArxivUrl("");
    setIsIngesting(false);
  };

  // Convert taxonomy to tree-compatible format
  const treeData = useMemo(() => toTreeData(taxonomy), [taxonomy]);

  // Find original node by name for details display
  const findNode = useCallback((tree: PaperNode, name: string): PaperNode | null => {
    if (tree.name === name) return tree;
    for (const child of tree.children || []) {
      const found = findNode(child, name);
      if (found) return found;
    }
    return null;
  }, []);

  const handleNodeClick = (nodeData: any) => {
    const node = findNode(taxonomy, nodeData.data.name);
    setSelectedNode(node);
    setActivePanel("details");
    setRepos([]);
    setReferences([]);
    setSimilarPapers([]);
  };

  const handleNodeRightClick = (event: React.MouseEvent, nodeData: any) => {
    event.preventDefault();
    const node = findNode(taxonomy, nodeData.data.name);
    if (node?.attributes?.arxivId) {
      setContextMenu({
        visible: true,
        x: event.clientX,
        y: event.clientY,
        node,
      });
    }
  };

  // Feature handlers
  const handleFindRepos = async (node: PaperNode) => {
    if (!node.attributes?.arxivId) return;
    setSelectedNode(node);
    setActivePanel("repos");
    setIsLoadingFeature(true);
    
    try {
      const res = await fetch("/api/repos/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          arxiv_id: node.attributes.arxivId,
          title: node.attributes.title,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setRepos(data.repos || []);
      }
    } catch (e) {
      console.error("Failed to fetch repos:", e);
    } finally {
      setIsLoadingFeature(false);
    }
  };

  const handleFetchReferences = async (node: PaperNode) => {
    if (!node.attributes?.arxivId) return;
    setSelectedNode(node);
    setActivePanel("references");
    setIsLoadingFeature(true);
    setRefExplanations({});
    
    try {
      const res = await fetch("/api/references/fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id: node.attributes.arxivId }),
      });
      if (res.ok) {
        const data = await res.json();
        setReferences(data.references || []);
        // Pre-populate cached explanations
        const cached: Record<number, string> = {};
        for (const ref of data.references || []) {
          if (ref.explanation) {
            cached[ref.id] = ref.explanation;
          }
        }
        setRefExplanations(cached);
      }
    } catch (e) {
      console.error("Failed to fetch references:", e);
    } finally {
      setIsLoadingFeature(false);
    }
  };

  const handleFindSimilar = async (node: PaperNode) => {
    if (!node.attributes?.arxivId) return;
    setSelectedNode(node);
    setActivePanel("similar");
    setIsLoadingFeature(true);
    
    try {
      const res = await fetch("/api/papers/similar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id: node.attributes.arxivId }),
      });
      if (res.ok) {
        const data = await res.json();
        setSimilarPapers(data.similar_papers || []);
      }
    } catch (e) {
      console.error("Failed to find similar papers:", e);
    } finally {
      setIsLoadingFeature(false);
    }
  };

  // Reference hover handler with debounce
  const handleRefHover = (ref: Reference) => {
    if (refExplanations[ref.id]) {
      setHoveredRefId(ref.id);
      return;
    }
    
    const debounceMs = uiConfig?.hover_debounce_ms || 500;
    
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
    
    hoverTimeoutRef.current = setTimeout(async () => {
      setHoveredRefId(ref.id);
      
      try {
        const res = await fetch("/api/references/explain", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            reference_id: ref.id,
            source_paper_title: selectedNode?.attributes?.title || "",
            cited_title: ref.cited_title,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          setRefExplanations((prev) => ({ ...prev, [ref.id]: data.explanation }));
        }
      } catch (e) {
        console.error("Failed to get explanation:", e);
      }
    }, debounceMs);
  };

  const handleRefLeave = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
    setHoveredRefId(null);
  };

  const handleAddSimilarPaper = async (paper: SimilarPaper) => {
    if (!paper.arxiv_id) return;
    setArxivUrl(paper.arxiv_id);
    setActivePanel("details");
  };

  const handleAddReference = async (ref: Reference) => {
    if (!ref.cited_arxiv_id) return;
    setArxivUrl(ref.cited_arxiv_id);
    setActivePanel("details");
  };

  const getStepIcon = (status: IngestionStep["status"]) => {
    switch (status) {
      case "pending": return "○";
      case "running": return "◐";
      case "done": return "●";
      case "error": return "✕";
    }
  };

  const getStepColor = (status: IngestionStep["status"]) => {
    switch (status) {
      case "pending": return "#999";
      case "running": return "#0070f3";
      case "done": return "#10b981";
      case "error": return "#ef4444";
    }
  };

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* Left panel: Tree visualization */}
      <div style={{ flex: 2, borderRight: "1px solid #e5e5e5", display: "flex", flexDirection: "column" }}>
        <div style={{ padding: "1rem", borderBottom: "1px solid #e5e5e5", backgroundColor: "#fafafa" }}>
          <h1 style={{ margin: 0, fontSize: "1.5rem", fontWeight: 600 }}>Paper Curator</h1>
          <p style={{ margin: "0.25rem 0 0", fontSize: "0.875rem", color: "#666" }}>
            {taxonomy.children?.length || 0} categories, {" "}
            {taxonomy.children?.reduce((acc, c) => acc + (c.children?.length || 0), 0) || 0} papers
          </p>
        </div>
        <div style={{ flex: 1, position: "relative" }}>
          {taxonomy.children && taxonomy.children.length > 0 ? (
            <Tree
              data={treeData}
              orientation="vertical"
              pathFunc="step"
              onNodeClick={handleNodeClick}
              translate={{ x: 300, y: 50 }}
              nodeSize={{ x: 220, y: 80 }}
              separation={{ siblings: 1.2, nonSiblings: 1.5 }}
            />
          ) : (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#999" }}>
              <p>No papers yet. Add one using the panel on the right.</p>
            </div>
          )}
        </div>
      </div>

      {/* Context Menu */}
      {contextMenu.visible && contextMenu.node && (
        <div
          style={{
            position: "fixed",
            top: contextMenu.y,
            left: contextMenu.x,
            backgroundColor: "white",
            border: "1px solid #e5e5e5",
            borderRadius: "4px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
            zIndex: 1000,
            minWidth: "180px",
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div
            style={{ padding: "0.5rem 1rem", cursor: "pointer", borderBottom: "1px solid #eee" }}
            onClick={() => { handleFindRepos(contextMenu.node!); setContextMenu({ ...contextMenu, visible: false }); }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f5f5f5")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "white")}
          >
            🔗 Find GitHub Repo
          </div>
          <div
            style={{ padding: "0.5rem 1rem", cursor: "pointer", borderBottom: "1px solid #eee" }}
            onClick={() => { handleFetchReferences(contextMenu.node!); setContextMenu({ ...contextMenu, visible: false }); }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f5f5f5")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "white")}
          >
            📚 Explain References
          </div>
          <div
            style={{ padding: "0.5rem 1rem", cursor: "pointer" }}
            onClick={() => { handleFindSimilar(contextMenu.node!); setContextMenu({ ...contextMenu, visible: false }); }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f5f5f5")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "white")}
          >
            🔍 Find Similar Papers
          </div>
        </div>
      )}

      {/* Right panel: Details and ingest */}
      <div style={{ flex: 1, padding: "1.5rem", display: "flex", flexDirection: "column", backgroundColor: "#fafafa", overflowY: "auto" }}>
        {/* Ingest section */}
        <div style={{ marginBottom: "2rem", backgroundColor: "white", padding: "1rem", borderRadius: "8px", border: "1px solid #e5e5e5" }}>
          <h2 style={{ marginTop: 0, fontSize: "1.125rem", fontWeight: 600 }}>Ingest Paper</h2>
          <p style={{ fontSize: "0.75rem", color: "#666", marginBottom: "0.75rem" }}>
            Enter an arXiv URL or ID. Right-click on papers for more options.
          </p>
          <input
            type="text"
            value={arxivUrl}
            onChange={(e) => setArxivUrl(e.target.value)}
            placeholder="arXiv URL or ID (e.g., 1706.03762)"
            disabled={isIngesting}
            style={{
              width: "100%",
              padding: "0.625rem",
              marginBottom: "0.75rem",
              boxSizing: "border-box",
              border: "1px solid #ddd",
              borderRadius: "4px",
              fontSize: "0.875rem",
            }}
          />
          <button
            onClick={handleIngest}
            disabled={isIngesting || !arxivUrl.trim()}
            style={{
              width: "100%",
              padding: "0.625rem",
              backgroundColor: isIngesting ? "#ccc" : "#0070f3",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: isIngesting ? "not-allowed" : "pointer",
              fontSize: "0.875rem",
              fontWeight: 500,
            }}
          >
            {isIngesting ? "Ingesting..." : "Ingest Paper"}
          </button>

          {/* Progress steps */}
          {steps.length > 0 && (
            <div style={{ marginTop: "1rem" }}>
              {steps.map((step, i) => (
                <div key={i} style={{ display: "flex", alignItems: "flex-start", marginBottom: "0.5rem" }}>
                  <span style={{ color: getStepColor(step.status), marginRight: "0.5rem", fontSize: "0.875rem" }}>
                    {getStepIcon(step.status)}
                  </span>
                  <div style={{ flex: 1 }}>
                    <span style={{ fontSize: "0.875rem", color: step.status === "error" ? "#ef4444" : "#333" }}>
                      {step.name}
                    </span>
                    {step.message && (
                      <p style={{ margin: "0.125rem 0 0", fontSize: "0.75rem", color: step.status === "error" ? "#ef4444" : "#666" }}>
                        {step.message}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Panel tabs */}
        {selectedNode && (
          <div style={{ display: "flex", marginBottom: "0.5rem", gap: "0.25rem" }}>
            {["details", "repos", "references", "similar"].map((panel) => (
              <button
                key={panel}
                onClick={() => setActivePanel(panel as any)}
                style={{
                  flex: 1,
                  padding: "0.5rem",
                  backgroundColor: activePanel === panel ? "#0070f3" : "#e5e5e5",
                  color: activePanel === panel ? "white" : "#333",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                  fontSize: "0.75rem",
                  textTransform: "capitalize",
                }}
              >
                {panel}
              </button>
            ))}
          </div>
        )}

        {/* Dynamic panel content */}
        <div style={{ flex: 1, backgroundColor: "white", padding: "1rem", borderRadius: "8px", border: "1px solid #e5e5e5", overflowY: "auto" }}>
          {/* Details Panel */}
          {activePanel === "details" && (
            <>
              <h2 style={{ marginTop: 0, fontSize: "1.125rem", fontWeight: 600 }}>Paper Details</h2>
              {selectedNode?.attributes ? (
                <div>
                  <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>{selectedNode.attributes.title || selectedNode.name}</h3>
                  {selectedNode.attributes.arxivId && (
                    <p style={{ fontSize: "0.875rem", color: "#666", margin: "0 0 0.5rem" }}>
                      <strong>arXiv:</strong>{" "}
                      <a href={`https://arxiv.org/abs/${selectedNode.attributes.arxivId}`} target="_blank" rel="noopener noreferrer" style={{ color: "#0070f3" }}>
                        {selectedNode.attributes.arxivId}
                      </a>
                    </p>
                  )}
                  {selectedNode.attributes.authors && (
                    <p style={{ fontSize: "0.875rem", color: "#666", margin: "0 0 0.5rem" }}>
                      <strong>Authors:</strong> {selectedNode.attributes.authors.slice(0, 3).join(", ")}
                      {selectedNode.attributes.authors.length > 3 && ` +${selectedNode.attributes.authors.length - 3} more`}
                    </p>
                  )}
                  {selectedNode.attributes.category && (
                    <p style={{ fontSize: "0.875rem", color: "#666", margin: "0 0 1rem" }}>
                      <strong>Category:</strong> {selectedNode.attributes.category}
                    </p>
                  )}
                  {selectedNode.attributes.summary && (
                    <div>
                      <h4 style={{ fontSize: "0.875rem", marginBottom: "0.5rem" }}>Summary</h4>
                      <p style={{ fontSize: "0.875rem", lineHeight: 1.6, color: "#333", whiteSpace: "pre-wrap" }}>
                        {selectedNode.attributes.summary}
                      </p>
                    </div>
                  )}
                </div>
              ) : selectedNode ? (
                <div>
                  <h3 style={{ fontSize: "1rem" }}>{selectedNode.name}</h3>
                  {selectedNode.children && (
                    <p style={{ fontSize: "0.875rem", color: "#666" }}>
                      {selectedNode.children.length} paper{selectedNode.children.length !== 1 ? "s" : ""} in this category
                    </p>
                  )}
                </div>
              ) : (
                <p style={{ color: "#999", fontSize: "0.875rem" }}>Click on a node in the tree to see details</p>
              )}
            </>
          )}

          {/* Repos Panel */}
          {activePanel === "repos" && (
            <>
              <h2 style={{ marginTop: 0, fontSize: "1.125rem", fontWeight: 600 }}>GitHub Repositories</h2>
              {isLoadingFeature ? (
                <p style={{ color: "#666" }}>Searching...</p>
              ) : repos.length > 0 ? (
                <div>
                  {repos.map((repo, i) => (
                    <div key={i} style={{ padding: "0.75rem", borderBottom: "1px solid #eee" }}>
                      <a href={repo.repo_url} target="_blank" rel="noopener noreferrer" style={{ color: "#0070f3", fontWeight: 500 }}>
                        {repo.repo_name}
                      </a>
                      <div style={{ fontSize: "0.75rem", color: "#666", marginTop: "0.25rem" }}>
                        {repo.is_official && <span style={{ color: "#10b981", marginRight: "0.5rem" }}>✓ Official</span>}
                        <span>⭐ {repo.stars || 0}</span>
                        <span style={{ marginLeft: "0.5rem" }}>via {repo.source}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: "#999", fontSize: "0.875rem" }}>No repositories found. Click "Find GitHub Repo" from the right-click menu.</p>
              )}
            </>
          )}

          {/* References Panel */}
          {activePanel === "references" && (
            <>
              <h2 style={{ marginTop: 0, fontSize: "1.125rem", fontWeight: 600 }}>References</h2>
              {isLoadingFeature ? (
                <p style={{ color: "#666" }}>Loading references...</p>
              ) : references.length > 0 ? (
                <div>
                  {references.map((ref) => (
                    <div
                      key={ref.id}
                      style={{ padding: "0.75rem", borderBottom: "1px solid #eee", position: "relative" }}
                      onMouseEnter={() => handleRefHover(ref)}
                      onMouseLeave={handleRefLeave}
                    >
                      <div style={{ fontSize: "0.875rem", fontWeight: 500 }}>{ref.cited_title}</div>
                      <div style={{ fontSize: "0.75rem", color: "#666", marginTop: "0.25rem" }}>
                        {ref.cited_authors?.slice(0, 2).join(", ")}
                        {ref.cited_authors && ref.cited_authors.length > 2 && " et al."}
                        {ref.cited_year && ` (${ref.cited_year})`}
                      </div>
                      {ref.cited_arxiv_id && (
                        <button
                          onClick={() => handleAddReference(ref)}
                          style={{
                            marginTop: "0.5rem",
                            padding: "0.25rem 0.5rem",
                            fontSize: "0.75rem",
                            backgroundColor: "#f0f0f0",
                            border: "none",
                            borderRadius: "4px",
                            cursor: "pointer",
                          }}
                        >
                          + Add to tree
                        </button>
                      )}
                      
                      {/* Hover tooltip */}
                      {hoveredRefId === ref.id && (
                        <div
                          style={{
                            position: "absolute",
                            left: "100%",
                            top: 0,
                            marginLeft: "0.5rem",
                            width: "300px",
                            padding: "0.75rem",
                            backgroundColor: "#333",
                            color: "white",
                            borderRadius: "4px",
                            fontSize: "0.75rem",
                            lineHeight: 1.5,
                            zIndex: 100,
                            boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
                          }}
                        >
                          {refExplanations[ref.id] || "Loading explanation..."}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: "#999", fontSize: "0.875rem" }}>No references loaded. Click "Explain References" from the right-click menu.</p>
              )}
            </>
          )}

          {/* Similar Papers Panel */}
          {activePanel === "similar" && (
            <>
              <h2 style={{ marginTop: 0, fontSize: "1.125rem", fontWeight: 600 }}>Similar Papers</h2>
              {isLoadingFeature ? (
                <p style={{ color: "#666" }}>Finding similar papers...</p>
              ) : similarPapers.length > 0 ? (
                <div>
                  {similarPapers.map((paper, i) => (
                    <div key={i} style={{ padding: "0.75rem", borderBottom: "1px solid #eee" }}>
                      <div style={{ fontSize: "0.875rem", fontWeight: 500 }}>{paper.title}</div>
                      <div style={{ fontSize: "0.75rem", color: "#666", marginTop: "0.25rem" }}>
                        Similarity: {((paper.similarity || 0) * 100).toFixed(1)}%
                      </div>
                      {paper.arxiv_id && (
                        <button
                          onClick={() => handleAddSimilarPaper(paper)}
                          style={{
                            marginTop: "0.5rem",
                            padding: "0.25rem 0.5rem",
                            fontSize: "0.75rem",
                            backgroundColor: "#f0f0f0",
                            border: "none",
                            borderRadius: "4px",
                            cursor: "pointer",
                          }}
                        >
                          + Add to tree
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: "#999", fontSize: "0.875rem" }}>No similar papers found. Click "Find Similar Papers" from the right-click menu.</p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
