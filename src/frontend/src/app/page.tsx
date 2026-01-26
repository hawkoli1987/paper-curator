"use client";

import { useState, useCallback, useMemo } from "react";
import dynamic from "next/dynamic";
import type { RawNodeDatum } from "react-d3-tree";

const Tree = dynamic(() => import("react-d3-tree"), { ssr: false });

interface PaperNode {
  name: string;
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
          categoryNode = { name: paper.category, children: [] };
          newTree.children = newTree.children || [];
          newTree.children.push(categoryNode);
        }
        categoryNode.children = categoryNode.children || [];
        categoryNode.children.push({
          name: paper.title.length > 40 ? paper.title.slice(0, 40) + "..." : paper.title,
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
    ]);

    let arxivId = "";
    let title = "";
    let authors: string[] = [];
    let abstract = "";
    let pdfPath = "";
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
  };

  const getStepIcon = (status: IngestionStep["status"]) => {
    switch (status) {
      case "pending":
        return "○";
      case "running":
        return "◐";
      case "done":
        return "●";
      case "error":
        return "✕";
    }
  };

  const getStepColor = (status: IngestionStep["status"]) => {
    switch (status) {
      case "pending":
        return "#999";
      case "running":
        return "#0070f3";
      case "done":
        return "#10b981";
      case "error":
        return "#ef4444";
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

      {/* Right panel: Details and ingest */}
      <div style={{ flex: 1, padding: "1.5rem", display: "flex", flexDirection: "column", backgroundColor: "#fafafa", overflowY: "auto" }}>
        {/* Ingest section */}
        <div style={{ marginBottom: "2rem", backgroundColor: "white", padding: "1rem", borderRadius: "8px", border: "1px solid #e5e5e5" }}>
          <h2 style={{ marginTop: 0, fontSize: "1.125rem", fontWeight: 600 }}>Ingest Paper</h2>
          <p style={{ fontSize: "0.75rem", color: "#666", marginBottom: "0.75rem" }}>
            Enter an arXiv URL or ID. The paper will be automatically classified by the LLM.
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

        {/* Node details section */}
        <div style={{ flex: 1, backgroundColor: "white", padding: "1rem", borderRadius: "8px", border: "1px solid #e5e5e5" }}>
          <h2 style={{ marginTop: 0, fontSize: "1.125rem", fontWeight: 600 }}>Paper Details</h2>
          {selectedNode?.attributes ? (
            <div>
              <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>{selectedNode.attributes.title || selectedNode.name}</h3>
              {selectedNode.attributes.arxivId && (
                <p style={{ fontSize: "0.875rem", color: "#666", margin: "0 0 0.5rem" }}>
                  <strong>arXiv:</strong>{" "}
                  <a
                    href={`https://arxiv.org/abs/${selectedNode.attributes.arxivId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: "#0070f3" }}
                  >
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
        </div>
      </div>
    </div>
  );
}
