"use client";

import { useState } from "react";
import dynamic from "next/dynamic";

// Dynamically import react-d3-tree to avoid SSR issues
const Tree = dynamic(() => import("react-d3-tree"), { ssr: false });

interface TaxonomyNode {
  name: string;
  children?: TaxonomyNode[];
  attributes?: {
    arxivId?: string;
    summary?: string;
  };
}

const initialTaxonomy: TaxonomyNode = {
  name: "Papers",
  children: [
    {
      name: "Machine Learning",
      children: [
        { name: "Transformers", attributes: { summary: "Placeholder" } },
        { name: "Reinforcement Learning", attributes: { summary: "Placeholder" } },
      ],
    },
    {
      name: "Natural Language Processing",
      children: [
        { name: "Large Language Models", attributes: { summary: "Placeholder" } },
      ],
    },
  ],
};

export default function Home() {
  const [arxivUrl, setArxivUrl] = useState("");
  const [status, setStatus] = useState("");
  const [selectedNode, setSelectedNode] = useState<TaxonomyNode | null>(null);
  const [taxonomy] = useState<TaxonomyNode>(initialTaxonomy);

  const handleIngest = async () => {
    if (!arxivUrl.trim()) {
      setStatus("Please enter an arXiv URL");
      return;
    }
    setStatus("Ingesting paper...");
    try {
      const response = await fetch("/api/arxiv/resolve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: arxivUrl }),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setStatus(`Resolved: ${data.title}`);
    } catch (error) {
      setStatus(`Error: ${error}`);
    }
  };

  const handleNodeClick = (nodeData: any) => {
    setSelectedNode(nodeData.data as TaxonomyNode);
  };

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* Left panel: Tree visualization */}
      <div style={{ flex: 2, borderRight: "1px solid #ccc", position: "relative" }}>
        <div style={{ padding: "1rem", borderBottom: "1px solid #ccc" }}>
          <h1 style={{ margin: 0, fontSize: "1.5rem" }}>Paper Curator</h1>
        </div>
        <div style={{ width: "100%", height: "calc(100% - 60px)" }}>
          <Tree
            data={taxonomy}
            orientation="vertical"
            pathFunc="step"
            onNodeClick={handleNodeClick}
            translate={{ x: 300, y: 50 }}
            nodeSize={{ x: 200, y: 100 }}
          />
        </div>
      </div>

      {/* Right panel: Details and ingest */}
      <div style={{ flex: 1, padding: "1rem", display: "flex", flexDirection: "column" }}>
        {/* Ingest section */}
        <div style={{ marginBottom: "2rem" }}>
          <h2 style={{ marginTop: 0 }}>Ingest Paper</h2>
          <input
            type="text"
            value={arxivUrl}
            onChange={(e) => setArxivUrl(e.target.value)}
            placeholder="Enter arXiv URL (e.g., https://arxiv.org/abs/1706.03762)"
            style={{
              width: "100%",
              padding: "0.5rem",
              marginBottom: "0.5rem",
              boxSizing: "border-box",
            }}
          />
          <button
            onClick={handleIngest}
            style={{
              padding: "0.5rem 1rem",
              backgroundColor: "#0070f3",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            Ingest
          </button>
          {status && (
            <p style={{ marginTop: "0.5rem", color: status.startsWith("Error") ? "red" : "green" }}>
              {status}
            </p>
          )}
        </div>

        {/* Node details section */}
        <div style={{ flex: 1 }}>
          <h2>Node Details</h2>
          {selectedNode ? (
            <div>
              <h3>{selectedNode.name}</h3>
              {selectedNode.attributes?.summary && (
                <div>
                  <h4>Summary</h4>
                  <p>{selectedNode.attributes.summary}</p>
                </div>
              )}
              {selectedNode.attributes?.arxivId && (
                <p>
                  <strong>arXiv ID:</strong> {selectedNode.attributes.arxivId}
                </p>
              )}
              {selectedNode.children && (
                <p>
                  <strong>Children:</strong> {selectedNode.children.length} nodes
                </p>
              )}
            </div>
          ) : (
            <p style={{ color: "#666" }}>Click on a node to see details</p>
          )}
        </div>
      </div>
    </div>
  );
}
