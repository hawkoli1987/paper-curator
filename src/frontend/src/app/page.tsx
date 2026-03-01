"use client";

import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import { hierarchy, tree as d3tree } from "d3-hierarchy";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from "@/components/ui/accordion";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Search, Settings } from "lucide-react";

interface PaperNode {
  name: string;
  node_id?: string;
  node_type?: string;
  paper_id?: number;
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
  arxiv_id?: string;
  title: string;
  similarity?: number;
  authors?: string[];
  abstract?: string;
  year?: number;
  citation_count?: number;
  url?: string;
  source?: string;
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
  tree_category_font_size: number;
  tree_paper_font_size: number;
}

interface TreeLayout {
  nodes: Array<{
    x: number;
    y: number;
    data: PaperNode;
    nodeId: string;
    isPaper: boolean;
  }>;
  links: Array<{
    source: { x: number; y: number; data: PaperNode };
    target: { x: number; y: number; data: PaperNode };
    isPaper: boolean;
  }>;
  width: number;
  height: number;
  offsetX: number;
  offsetY: number;
  linesById: Record<string, string[]>;
  dimsById: Record<string, { width: number; height: number; lineHeight: number; paddingY: number }>;
}

// Topic Query Interfaces
interface TopicPaper {
  paper_id: number;
  arxiv_id: string;
  title: string;
  year?: number;
  similarity: number;
  selected?: boolean;
}

interface Topic {
  id: number;
  name: string;
  topic_query: string;
  paper_count: number;
  query_count: number;
  created_at: string;
  papers?: TopicPaper[];
  queries?: TopicQuery[];
}

interface TopicQuery {
  id: number;
  question: string;
  answer: string;
  paper_responses?: Array<{
    paper_id: number;
    title: string;
    response: string;
    success: boolean;
  }>;
  created_at: string;
}

// Component to render formatted text with LaTeX, bold, and lists
function FormattedText({ text, className }: { text: string; className?: string }) {
  // Parse text into segments: LaTeX (inline/block), bold, and plain text
  const parseSegment = (segment: string): React.ReactNode[] => {
    const result: React.ReactNode[] = [];
    let remaining = segment;
    let keyCounter = 0;
    
    while (remaining.length > 0) {
      // Find the earliest match of any pattern
      const patterns = [
        { regex: /\\\(([\s\S]*?)\\\)/, type: 'inline-math' },
        { regex: /\\\[([\s\S]*?)\\\]/, type: 'block-math' },
        { regex: /\$([^$]+)\$/, type: 'inline-math-dollar' },
        { regex: /\*\*([^*]+)\*\*/, type: 'bold' },
      ];
      
      let earliestMatch: { index: number; length: number; type: string; content: string } | null = null;
      
      for (const pattern of patterns) {
        const match = pattern.regex.exec(remaining);
        if (match && (earliestMatch === null || match.index < earliestMatch.index)) {
          earliestMatch = {
            index: match.index,
            length: match[0].length,
            type: pattern.type,
            content: match[1],
          };
        }
      }
      
      if (earliestMatch) {
        // Add text before the match
        if (earliestMatch.index > 0) {
          result.push(<span key={keyCounter++}>{remaining.substring(0, earliestMatch.index)}</span>);
        }
        
        // Add the matched element
        if (earliestMatch.type === 'inline-math' || earliestMatch.type === 'inline-math-dollar') {
          result.push(<InlineMath key={keyCounter++} math={earliestMatch.content} />);
        } else if (earliestMatch.type === 'block-math') {
          result.push(<BlockMath key={keyCounter++} math={earliestMatch.content} />);
        } else if (earliestMatch.type === 'bold') {
          result.push(<strong key={keyCounter++} className="font-semibold text-gray-900">{earliestMatch.content}</strong>);
        }
        
        remaining = remaining.substring(earliestMatch.index + earliestMatch.length);
      } else {
        // No more patterns found, add remaining text
        result.push(<span key={keyCounter++}>{remaining}</span>);
        break;
      }
    }
    
    return result;
  };
  
  // Helper to render a list
  const renderList = (list: { type: 'ol' | 'ul'; items: React.ReactNode[] }, key: string) => {
    if (list.type === 'ol') {
      return <ol key={key} className="list-decimal list-inside ml-4 space-y-1">{list.items}</ol>;
    }
    return <ul key={key} className="list-disc list-inside ml-4 space-y-1">{list.items}</ul>;
  };
  
  // Split text into lines and process each
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];
  let currentList: { type: 'ol' | 'ul'; items: React.ReactNode[] } | null = null;
  
  lines.forEach((line, lineIdx) => {
    const trimmedLine = line.trim();
    
    // Check for numbered list item (1., 2., etc.)
    const numberedMatch = trimmedLine.match(/^(\d+)\.\s+(.*)$/);
    // Check for bullet point (-, •, *)
    const bulletMatch = trimmedLine.match(/^[-•*]\s+(.*)$/);
    
    if (numberedMatch) {
      if (!currentList || currentList.type !== 'ol') {
        // Start new ordered list
        if (currentList) {
          elements.push(renderList(currentList, `list-${lineIdx}`));
        }
        currentList = { type: 'ol', items: [] };
      }
      currentList.items.push(
        <li key={lineIdx} className="text-gray-600">{parseSegment(numberedMatch[2])}</li>
      );
    } else if (bulletMatch) {
      if (!currentList || currentList.type !== 'ul') {
        // Start new unordered list
        if (currentList) {
          elements.push(renderList(currentList, `list-${lineIdx}`));
        }
        currentList = { type: 'ul', items: [] };
      }
      currentList.items.push(
        <li key={lineIdx} className="text-gray-600">{parseSegment(bulletMatch[1])}</li>
      );
    } else {
      // Not a list item - flush any pending list
      if (currentList) {
        elements.push(renderList(currentList, `list-${lineIdx}`));
        currentList = null;
      }
      
      // Add the line (with line break if not last)
      if (trimmedLine.length > 0) {
        elements.push(
          <span key={lineIdx} className="block">
            {parseSegment(line)}
          </span>
        );
      } else if (lineIdx < lines.length - 1) {
        // Empty line = paragraph break
        elements.push(<span key={lineIdx} className="block h-2" />);
      }
    }
  });
  
  // Flush any remaining list
  if (currentList) {
    elements.push(renderList(currentList, "list-final"));
  }
  
  return <div className={className}>{elements}</div>;
}

// Legacy component for backward compatibility
function TextWithMath({ text, style }: { text: string; style?: React.CSSProperties }) {
  return <FormattedText text={text} className="" />;
}

// Note: CollapsibleSection replaced with Shadcn/ui Accordion component

const initialTaxonomy: PaperNode = {
  name: "AI Papers",
  children: [],
};

export default function Home() {
  const [unifiedIngestInput, setUnifiedIngestInput] = useState("");
  const [taxonomy, setTaxonomy] = useState<PaperNode>(initialTaxonomy);
  const [selectedNode, setSelectedNode] = useState<PaperNode | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isLoadingTree, setIsLoadingTree] = useState(true); // Start with loading state
  
  // Right panel font size (in pixels) - default 14
  const [panelFontSize, setPanelFontSize] = useState(14);
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
  const [activePanel, setActivePanel] = useState<"explorer" | "repos" | "references" | "similar" | "query" | "topic-query">("explorer");
  const [repos, setRepos] = useState<RepoResult[]>([]);
  const [queryInput, setQueryInput] = useState("");
  const [queryAnswer, setQueryAnswer] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);
  const [isReabbreviating, setIsReabbreviating] = useState(false);
  const [isLoadingStructured, setIsLoadingStructured] = useState(false);
  const [structuredAnalysis, setStructuredAnalysis] = useState<{
    components: string[];
    sections: Array<{
      component: string;
      steps: string;
      benefits: string;
      rationale: string;
      results: string;
    }>;
  } | null>(null);
  const [references, setReferences] = useState<Reference[]>([]);
  const [similarPapers, setSimilarPapers] = useState<SimilarPaper[]>([]);
  const [queryHistory, setQueryHistory] = useState<Array<{id: number; question: string; answer: string; created_at: string}>>([]);
  const [isLoadingFeature, setIsLoadingFeature] = useState(false);
  const [isLoadingCachedData, setIsLoadingCachedData] = useState(false);
  const [hoveredRefId, setHoveredRefId] = useState<number | null>(null);
  const [refExplanations, setRefExplanations] = useState<Record<number, string>>({});
  const [featureLog, setFeatureLog] = useState<string[]>([]);
  const [ingestLog, setIngestLog] = useState<string[]>([]);
  
  // Unified ingest state (replaces separate arxivUrl and batchDirectory)
  const [isBatchIngesting, setIsBatchIngesting] = useState(false);
  const [batchResults, setBatchResults] = useState<{
    total: number;
    success: number;
    skipped: number;
    errors: number;
    results: Array<{ file: string; status: string; reason?: string; title?: string; category?: string }>;
    progress_log?: string[];
  } | null>(null);
  
  // Slack credential prompt state
  const [showSlackCredentialPrompt, setShowSlackCredentialPrompt] = useState(false);
  const [pendingSlackChannel, setPendingSlackChannel] = useState("");
  const [slackToken, setSlackToken] = useState("");
  
  // Categorize state
  const [isCategorizing, setIsCategorizing] = useState(false);
  const [categorizeResult, setCategorizeResult] = useState<{
    message: string;
    mode: string;
    papers_classified?: number;
    clusters_created?: number;
    recategorized?: number;
    nodes_named: number;
  } | null>(null);
  
  // Collapsible sections
  
  // Query selection for merge feature
  const [selectedQueryIds, setSelectedQueryIds] = useState<Set<number>>(new Set());
  const [isMergingQueries, setIsMergingQueries] = useState(false);
  const [isDedupingSummary, setIsDedupingSummary] = useState(false);
  
  // Tree diagram state - temporarily disabled during rebuild
  // const [collapsedCategories, setCollapsedCategories] = useState<Set<string>>(new Set());
  // const [nodes, setNodes, onNodesChange] = useNodesState([]);
  // const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  
  // Phase 3: Navigation & Interaction states (only search kept)
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<PaperNode[]>([]);
  const [isSearchFocused, setIsSearchFocused] = useState(false);
  const [searchMode, setSearchMode] = useState<"paper" | "category" | "topic">("paper");
  
  // Topic Query State
  const [topicList, setTopicList] = useState<Topic[]>([]);
  const [topicSearchResults, setTopicSearchResults] = useState<TopicPaper[]>([]);
  const [topicPool, setTopicPool] = useState<TopicPaper[]>([]);
  const [currentTopic, setCurrentTopic] = useState<Topic | null>(null);
  const [topicSearchOffset, setTopicSearchOffset] = useState(0);
  const [showTopicDialog, setShowTopicDialog] = useState(false);
  const [topicDialogMode, setTopicDialogMode] = useState<"check" | "select" | "create" | "building">("check");
  const [existingTopics, setExistingTopics] = useState<Topic[]>([]);
  const [topicNamePrefix, setTopicNamePrefix] = useState("");
  const [currentTopicQuery, setCurrentTopicQuery] = useState("");
  const [topicQueryInput, setTopicQueryInput] = useState("");
  const [isLoadingTopicSearch, setIsLoadingTopicSearch] = useState(false);
  const [isLoadingTopicQuery, setIsLoadingTopicQuery] = useState(false);
  const [topicEmbedding, setTopicEmbedding] = useState<number[] | null>(null);
  const [expandedTopicId, setExpandedTopicId] = useState<number | null>(null);
  
  // Parent map for ancestor chain navigation (node_id -> parent node)
  const [parentMap, setParentMap] = useState<Map<string, PaperNode>>(new Map());
  
  // Pulse animation for centered node
  const [pulsingNodeId, setPulsingNodeId] = useState<string | null>(null);
  
  // Phase 4: Responsive Design & Polish states
  const [windowWidth, setWindowWidth] = useState(typeof window !== 'undefined' ? window.innerWidth : 1024);
  const [rightPanelWidth, setRightPanelWidth] = useState(50); // Percentage
  const [isRightCollapsed, setIsRightCollapsed] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState<"tree" | "details" | null>(null);
  
  // Tree zoom and navigation
  const [zoomLevel, setZoomLevel] = useState(1);
  const [viewportPos, setViewportPos] = useState({ x: 0, y: 0 }); // For minimap
  const treeContainerRef = useRef<HTMLDivElement>(null);
  const [isDebugPanelOpen, setIsDebugPanelOpen] = useState(true); // Debug log shown by default
  const [hasAutoZoomed, setHasAutoZoomed] = useState(false); // Track if initial auto-zoom done
  
  // Settings modal state
  const [showSettings, setShowSettings] = useState(false);
  const [settingsData, setSettingsData] = useState<any>(null);
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [settingsWarnings, setSettingsWarnings] = useState<string[]>([]);
  const [settingsSaving, setSettingsSaving] = useState(false);
  
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Helper to add to feature log
  const logFeature = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setFeatureLog((prev) => [...prev, `[${timestamp}] ${message}`]);
  };
  
  const clearFeatureLog = () => setFeatureLog([]);
  
  // Helper to add to ingest log
  const logIngest = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setIngestLog((prev) => [...prev, `[${timestamp}] ${message}`]);
  };
  
  const clearIngestLog = () => setIngestLog([]);

  // Load config and tree on mount
  useEffect(() => {
    const loadInitialData = async () => {
      // Load UI config
      try {
        const configRes = await fetch("/api/ui-config");
        if (configRes.ok) {
          const config = await configRes.json();
          setUiConfig(config);
        }
      } catch (e) {
        console.error("Failed to load UI config:", e);
      }
      
      // Load tree from database
      setIsLoadingTree(true);
      try {
        const treeRes = await fetch("/api/tree");
        if (treeRes.ok) {
          const treeData = await treeRes.json();
          if (treeData.children && treeData.children.length > 0) {
            setTaxonomy(treeData);
            // Default to showing root node details
            setSelectedNode(treeData);
            // Reset auto-zoom flag to trigger fit-to-view on initial load
            setHasAutoZoomed(false);
          }
        }
      } catch (e) {
        console.error("Failed to load tree:", e);
      } finally {
        setIsLoadingTree(false);
      }
      
      // Load topics
      try {
        const topicsRes = await fetch("/api/topic/list");
        if (topicsRes.ok) {
          const data = await topicsRes.json();
          setTopicList(data.topics || []);
        }
      } catch (e) {
        console.error("Failed to load topics:", e);
      }
    };
    
    loadInitialData();
  }, []);

  // Build parent map whenever taxonomy changes (for ancestor chain navigation)
  useEffect(() => {
    const newParentMap = new Map<string, PaperNode>();
    
    const buildParentMap = (node: PaperNode, parent: PaperNode | null) => {
      const nodeId = node.node_id || node.name;
      if (parent) {
        newParentMap.set(nodeId, parent);
      }
      node.children?.forEach(child => buildParentMap(child, node));
    };
    
    buildParentMap(taxonomy, null);
    setParentMap(newParentMap);
  }, [taxonomy]);
  
  // Get ancestor chain for a node (from root to parent, not including the node itself)
  const getAncestorChain = useCallback((node: PaperNode): PaperNode[] => {
    const ancestors: PaperNode[] = [];
    const nodeId = node.node_id || node.name;
    let current = parentMap.get(nodeId);
    
    while (current) {
      ancestors.unshift(current); // Add to beginning for root-first order
      const currentId = current.node_id || current.name;
      current = parentMap.get(currentId);
    }
    
    return ancestors;
  }, [parentMap]);
  
  // Center tree view on a specific node with zoom to absolute 85% and pulse animation
  const centerOnNode = useCallback((nodeId: string) => {
    if (!treeContainerRef.current) return;
    
    const container = treeContainerRef.current;
    const nodeElement = container.querySelector(`[data-node-id="${nodeId}"]`) as HTMLElement | null;
    
    if (nodeElement) {
      const containerRect = container.getBoundingClientRect();
      const nodeRect = nodeElement.getBoundingClientRect();
      
      // Calculate node center in current viewport
      const nodeCenterX = nodeRect.left - containerRect.left + nodeRect.width / 2;
      const nodeCenterY = nodeRect.top - containerRect.top + nodeRect.height / 2;
      
      // Set zoom to absolute 85% (0.85) and center on node
      setZoomLevel((currentZoom) => {
        const newZoom = 0.85; // Absolute 85% zoom
        
        // Calculate node position in SVG coordinates
        const svgNodeX = (container.scrollLeft + nodeCenterX) / currentZoom;
        const svgNodeY = (container.scrollTop + nodeCenterY) / currentZoom;
        
        // Calculate scroll position to center the node at new zoom
        const newScrollLeft = svgNodeX * newZoom - container.clientWidth / 2;
        const newScrollTop = svgNodeY * newZoom - container.clientHeight / 2;
        
        // Apply scroll after DOM update
        setTimeout(() => {
          container.scrollTo({
            left: Math.max(0, newScrollLeft),
            top: Math.max(0, newScrollTop),
            behavior: 'smooth'
          });
        }, 50);
        
        return newZoom;
      });
      
      // Trigger pulse animation
      setPulsingNodeId(nodeId);
      setTimeout(() => setPulsingNodeId(null), 1500);
    }
  }, []);

  // Close context menu on click outside
  useEffect(() => {
    const handleClick = () => setContextMenu((prev) => ({ ...prev, visible: false }));
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, []);

  // Prevent browser zoom when mouse is inside tree container
  // Uses native event listener with passive:false to fully capture wheel events
  useEffect(() => {
    const container = treeContainerRef.current;
    if (!container) return;
    
    const handleWheel = (e: WheelEvent) => {
      // Prevent browser zoom when Ctrl/Cmd + scroll
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
      }
    };
    
    // passive: false is required to allow preventDefault
    container.addEventListener('wheel', handleWheel, { passive: false });
    
    return () => {
      container.removeEventListener('wheel', handleWheel);
    };
  }, []);

  const updateStep = (index: number, update: Partial<IngestionStep>) => {
    setSteps((prev) => prev.map((s, i) => (i === index ? { ...s, ...update } : s)));
  };

  // Get existing categories from the tree

  const _doCategorize = async (full: boolean) => {
    setIsCategorizing(true);
    setCategorizeResult(null);
    clearIngestLog();
    const label = full ? "full rebuild" : "partial recategorize";
    logIngest(`Starting ${label}...`);

    try {
      const url = full ? "/api/papers/categorize?full=true" : "/api/papers/categorize";
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      if (!res.ok) {
        const errText = await res.text();
        logIngest(`Error: HTTP ${res.status}`);
        try {
          const errJson = JSON.parse(errText);
          logIngest(`Details: ${errJson.detail || errText}`);
        } catch {
          logIngest(`Details: ${errText.slice(0, 200)}`);
        }
        return;
      }

      const data = await res.json();
      setCategorizeResult(data);
      logIngest(`✓ ${data.message}`);
      if (data.papers_classified !== undefined) logIngest(`  Papers classified: ${data.papers_classified}`);
      if (data.clusters_created !== undefined) logIngest(`  Clusters created: ${data.clusters_created}`);
      if (data.recategorized !== undefined) logIngest(`  Nodes recategorized: ${data.recategorized}`);
      logIngest(`  Nodes named: ${data.nodes_named}`);

      // Refresh tree
      const treeRes = await fetch("/api/tree");
      if (treeRes.ok) {
        const treeData = await treeRes.json();
        setTaxonomy(treeData);
        logIngest("Tree refreshed");
      }
    } catch (e) {
      logIngest(`Error: ${e}`);
    } finally {
      setIsCategorizing(false);
      logIngest("Done");
    }
  };

  const handleCategorizePartial = () => _doCategorize(false);

  const handleCategorizeFull = () => {
    if (!window.confirm(
      "Full rebuild re-clusters ALL papers from scratch and may take 30+ minutes.\n\nProceed?"
    )) return;
    _doCategorize(true);
  };

  // =========================================================================
  // Settings Handlers
  // =========================================================================
  
  const loadSettings = async () => {
    setSettingsLoading(true);
    try {
      const res = await fetch("/api/config");
      if (res.ok) {
        const data = await res.json();
        setSettingsData(data);
      }
    } catch (e) {
      console.error("Failed to load settings:", e);
    } finally {
      setSettingsLoading(false);
    }
  };
  
  const handleOpenSettings = () => {
    setShowSettings(true);
    setSettingsWarnings([]);
    loadSettings();
  };
  
  const handleSaveSettings = async (updates: Record<string, any>) => {
    setSettingsSaving(true);
    setSettingsWarnings([]);
    try {
      const res = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings: updates }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.warnings && data.warnings.length > 0) {
          setSettingsWarnings(data.warnings);
        }
        // Reload settings to show updated values
        await loadSettings();
        if (data.errors && data.errors.length > 0) {
          alert("Some settings failed to save: " + data.errors.join(", "));
        }
      }
    } catch (e) {
      console.error("Failed to save settings:", e);
      alert("Failed to save settings");
    } finally {
      setSettingsSaving(false);
    }
  };
  
  const handleResetSettings = async () => {
    if (!confirm("Reset all settings to defaults from config.yaml?")) return;
    
    setSettingsSaving(true);
    try {
      const res = await fetch("/api/config/reset", { method: "POST" });
      if (res.ok) {
        await loadSettings();
        setSettingsWarnings([]);
      }
    } catch (e) {
      console.error("Failed to reset settings:", e);
    } finally {
      setSettingsSaving(false);
    }
  };

  // =========================================================================
  // Topic Query Handlers
  // =========================================================================
  
  // Handle topic search when user presses Enter in search box with topic mode
  const handleTopicSearch = async (topic: string) => {
    if (!topic.trim()) return;
    
    setCurrentTopicQuery(topic);
    setIsLoadingTopicSearch(true);
    
    try {
      // Check if topics exist for this query
      const checkRes = await fetch(`/api/topic/check?topic_query=${encodeURIComponent(topic)}`);
      if (checkRes.ok) {
        const data = await checkRes.json();
        if (data.exists && data.topics.length > 0) {
          // Show dialog to choose resume or create new
          setExistingTopics(data.topics);
          setTopicDialogMode("select");
          setShowTopicDialog(true);
        } else {
          // No existing topics, start creating new one
          await startNewTopic(topic);
        }
      }
    } catch (e) {
      console.error("Topic check failed:", e);
    } finally {
      setIsLoadingTopicSearch(false);
    }
  };
  
  // Start a new topic with paper search
  const startNewTopic = async (topic: string, prefix: string = "") => {
    setIsLoadingTopicSearch(true);
    setTopicDialogMode("building");
    setShowTopicDialog(true);
    
    try {
      // Search for papers matching the topic
      const searchRes = await fetch("/api/topic/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic, offset: 0, limit: 10 }),
      });
      
      if (searchRes.ok) {
        const data = await searchRes.json();
        setTopicSearchResults(data.papers.map((p: TopicPaper) => ({ ...p, selected: false })));
        setTopicEmbedding(data.topic_embedding);
        setTopicSearchOffset(10);
        
        // Create the topic in DB
        const topicName = prefix ? `${prefix}: ${topic}` : topic;
        const createRes = await fetch("/api/topic/create", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: topicName, topic_query: topic }),
        });
        
        if (createRes.ok) {
          const topicData = await createRes.json();
          setCurrentTopic({ 
            id: topicData.topic_id, 
            name: topicData.name, 
            topic_query: topic,
            paper_count: 0,
            query_count: 0,
            created_at: new Date().toISOString(),
          });
          setTopicPool([]);
        }
      }
    } catch (e) {
      console.error("Topic search failed:", e);
    } finally {
      setIsLoadingTopicSearch(false);
    }
  };
  
  // Resume an existing topic
  const resumeTopic = async (topic: Topic) => {
    setIsLoadingTopicSearch(true);
    
    try {
      const res = await fetch(`/api/topic/${topic.id}`);
      if (res.ok) {
        const data = await res.json();
        setCurrentTopic(data);
        setTopicPool(data.papers || []);
        setShowTopicDialog(false);
        // Switch to topic-query tab
        setActivePanel("topic-query" as any);
      }
    } catch (e) {
      console.error("Failed to load topic:", e);
    } finally {
      setIsLoadingTopicSearch(false);
    }
  };
  
  // Load more papers for topic selection
  const loadMoreTopicPapers = async () => {
    if (!currentTopicQuery || !topicEmbedding) return;
    
    setIsLoadingTopicSearch(true);
    
    try {
      const excludeIds = [...topicPool.map(p => p.paper_id), ...topicSearchResults.map(p => p.paper_id)];
      const searchRes = await fetch("/api/topic/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          topic: currentTopicQuery, 
          offset: topicSearchOffset, 
          limit: 10,
          exclude_paper_ids: excludeIds,
        }),
      });
      
      if (searchRes.ok) {
        const data = await searchRes.json();
        setTopicSearchResults(data.papers.map((p: TopicPaper) => ({ ...p, selected: false })));
        setTopicSearchOffset(topicSearchOffset + 10);
      }
    } catch (e) {
      console.error("Load more failed:", e);
    } finally {
      setIsLoadingTopicSearch(false);
    }
  };
  
  // Add selected papers to pool
  const addSelectedPapersToPool = async () => {
    if (!currentTopic) return;
    
    const selected = topicSearchResults.filter(p => p.selected);
    if (selected.length === 0) return;
    
    try {
      const res = await fetch(`/api/topic/${currentTopic.id}/papers`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          paper_ids: selected.map(p => p.paper_id),
          similarity_scores: selected.map(p => p.similarity),
        }),
      });
      
      if (res.ok) {
        // Add to local pool
        setTopicPool(prev => [...prev, ...selected]);
        // Clear selection and load more
        setTopicSearchResults([]);
        await loadMoreTopicPapers();
      }
    } catch (e) {
      console.error("Failed to add papers:", e);
    }
  };
  
  // Remove a paper from the pool
  const removePaperFromPool = async (paperId: number) => {
    if (!currentTopic) return;
    
    try {
      await fetch(`/api/topic/${currentTopic.id}/papers/${paperId}`, {
        method: "DELETE",
      });
      setTopicPool(prev => prev.filter(p => p.paper_id !== paperId));
    } catch (e) {
      console.error("Failed to remove paper:", e);
    }
  };
  
  // Finish paper selection and go to query mode
  const finishPaperSelection = () => {
    setShowTopicDialog(false);
    setTopicSearchResults([]);
    setActivePanel("topic-query" as any);
    // Refresh topic list
    loadTopicList();
  };
  
  // Load all topics
  const loadTopicList = async () => {
    try {
      const res = await fetch("/api/topic/list");
      if (res.ok) {
        const data = await res.json();
        setTopicList(data.topics);
      }
    } catch (e) {
      console.error("Failed to load topics:", e);
    }
  };
  
  // Delete a topic
  const deleteTopic = async (topicId: number) => {
    if (!confirm("Delete this topic and all its queries?")) return;
    
    try {
      await fetch(`/api/topic/${topicId}`, { method: "DELETE" });
      setTopicList(prev => prev.filter(t => t.id !== topicId));
      if (currentTopic?.id === topicId) {
        setCurrentTopic(null);
        setTopicPool([]);
      }
    } catch (e) {
      console.error("Failed to delete topic:", e);
    }
  };
  
  // Query the current topic
  const handleTopicQuerySubmit = async () => {
    if (!currentTopic || !topicQueryInput.trim()) return;
    
    setIsLoadingTopicQuery(true);
    
    try {
      const res = await fetch(`/api/topic/${currentTopic.id}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: topicQueryInput }),
      });
      
      if (res.ok) {
        const data = await res.json();
        // Refresh topic to get updated queries
        const topicRes = await fetch(`/api/topic/${currentTopic.id}`);
        if (topicRes.ok) {
          const topicData = await topicRes.json();
          setCurrentTopic(topicData);
          setTopicPool(topicData.papers || []);
        }
        setTopicQueryInput("");
        // Refresh topic list
        loadTopicList();
      } else {
        const error = await res.json().catch(() => ({}));
        alert(`Query failed: ${error.detail || res.statusText}`);
      }
    } catch (e) {
      console.error("Topic query failed:", e);
      alert(`Error: ${e}`);
    } finally {
      setIsLoadingTopicQuery(false);
    }
  };

  // Unified ingest handler with auto-detection
  const handleUnifiedIngest = async () => {
    if (!unifiedIngestInput.trim()) return;
    
    const input = unifiedIngestInput.trim();
    
    // Detect input type
    const isUrl = input.startsWith("http://") || input.startsWith("https://") || input.includes("arxiv.org");
    const isArxivId = /^\d{4}\.\d{4,5}(v\d+)?$/.test(input) || /^arxiv:\/\/(abs|pdf)\/\d{4}\.\d{4,5}(v\d+)?/.test(input);
    // Slack channel: starts with #, contains slack.com, or is a Slack channel ID (starts with C followed by alphanumeric)
    const isSlackChannel = input.startsWith("#") || input.includes("slack.com") || /^C[A-Z0-9]{8,}$/.test(input);
    
    // Check if it's a directory path (local folder)
    // Simple heuristic: if it starts with / or contains \ and doesn't look like a URL
    const isDirectory = (input.startsWith("/") || input.includes("\\")) && !isUrl && !isArxivId && !isSlackChannel;
    
    if (isSlackChannel) {
      // Show credential prompt for Slack channel
      setPendingSlackChannel(input);
      setShowSlackCredentialPrompt(true);
      return;
    }
    
    if (isDirectory) {
      // Batch ingest
      setIsBatchIngesting(true);
      setBatchResults(null);
      clearIngestLog();
      logIngest(`Starting batch ingest from: ${input}`);
      
      try {
        const res = await fetch("/api/papers/batch-ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ directory: input, skip_existing: true }),
        });
        
        if (!res.ok) {
          const errText = await res.text();
          logIngest(`Error: HTTP ${res.status}`);
          try {
            const errJson = JSON.parse(errText);
            logIngest(`Details: ${errJson.detail || errText}`);
          } catch {
            logIngest(`Details: ${errText.slice(0, 200)}`);
          }
          setIsBatchIngesting(false);
          return;
        }
        
        const data = await res.json();
        setBatchResults(data);
        logIngest(`Completed: ${data.success} success, ${data.skipped} skipped, ${data.errors} errors`);
        
        // Refresh tree
        const treeRes = await fetch("/api/tree");
        if (treeRes.ok) {
          const treeData = await treeRes.json();
          setTaxonomy(treeData);
          logIngest("Tree refreshed");
        }
      } catch (err) {
        logIngest(`Error: ${err}`);
      } finally {
        setIsBatchIngesting(false);
        logIngest("Done");
      }
    } else {
      // Single ingest (URL or arXiv ID)
      setIsIngesting(true);
      clearIngestLog();
      logIngest(`Starting ingestion for: ${input}`);
      setSteps([
        { name: "Resolve arXiv", status: "pending" },
        { name: "Download PDF", status: "pending" },
        { name: "Extract text", status: "pending" },
        { name: "Abbreviate (LLM)", status: "pending" },
        { name: "Summarize (LLM)", status: "pending" },
        { name: "Save", status: "pending" },
      ]);

      let arxivId = "";
      let title = "";
      let authors: string[] = [];
      let abstract = "";
      let pdfPath = "";
      let latexPath = "";
      let pdfUrl = "";
      let published = "";
      let abbreviation = "";
      let summary = "";

      // Step 1: Resolve
      updateStep(0, { status: "running" });
      logIngest("Resolving arXiv metadata...");
      // Detect if input is URL or plain arXiv ID
      const inputValue = input;
      const isUrl = inputValue.includes("arxiv.org") || inputValue.startsWith("http");
    const resolveRes = await fetch("/api/arxiv/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(isUrl ? { url: inputValue } : { arxiv_id: inputValue }),
    });
    if (!resolveRes.ok) {
      const errText = await resolveRes.text();
      logIngest(`Error: HTTP ${resolveRes.status} - ${errText}`);
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
    logIngest(`Found: ${title}`);
    logIngest(`arXiv ID: ${arxivId}`);
    
    // Check for duplicate
    const checkDupRes = await fetch("/api/papers/cached-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arxiv_id: arxivId }),
    });
    if (checkDupRes.ok) {
      logIngest(`⚠️ Paper already exists in database!`);
      logIngest(`Skipping ingestion - paper ${arxivId} was previously ingested.`);
      updateStep(0, { status: "done", message: `${title} (already exists)` });
      // Mark remaining steps as skipped
      for (let i = 1; i < 6; i++) {
        updateStep(i, { status: "done", message: "Skipped (duplicate)" });
      }
      // Refresh tree to ensure search/index is up to date
      const treeRes = await fetch("/api/tree");
      if (treeRes.ok) {
        const treeData = await treeRes.json();
        setTaxonomy(treeData);
        logIngest("Tree refreshed");
      }
      setIsIngesting(false);
      return;
    }
    
    updateStep(0, { status: "done", message: title });

    // Step 2: Download
    updateStep(1, { status: "running" });
    logIngest("Downloading PDF...");
    const downloadRes = await fetch("/api/arxiv/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arxiv_id: arxivId }),
    });
    if (!downloadRes.ok) {
      const errText = await downloadRes.text();
      logIngest(`Error: HTTP ${downloadRes.status} - ${errText}`);
      updateStep(1, { status: "error", message: `HTTP ${downloadRes.status}` });
      setIsIngesting(false);
      return;
    }
    const downloadData = await downloadRes.json();
    pdfPath = downloadData.pdf_path;
    latexPath = downloadData.latex_path;
    logIngest(`PDF saved to: ${pdfPath}`);
    updateStep(1, { status: "done", message: "PDF downloaded" });

    // Steps 3-4: Extract + Abbreviate in PARALLEL
    // These are independent: extract uses PDF, abbreviate uses title
    logIngest("Running Extract + Abbreviate in parallel...");
    updateStep(2, { status: "running" });
    updateStep(3, { status: "running", message: "Creating short name..." });

    const [extractRes, abbreviateRes] = await Promise.all([
      fetch("/api/pdf/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pdf_path: pdfPath }),
      }),
      fetch("/api/abbreviate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      }),
    ]);

    // Handle Extract result
    if (!extractRes.ok) {
      updateStep(2, { status: "error", message: `HTTP ${extractRes.status}` });
      setIsIngesting(false);
      return;
    }
    updateStep(2, { status: "done", message: "Text extracted" });

    // Handle Abbreviate result
    if (abbreviateRes.ok) {
      const abbreviateData = await abbreviateRes.json();
      abbreviation = abbreviateData.abbreviation;
      updateStep(3, { status: "done", message: abbreviation });
    } else {
      // Fallback to truncated title
      abbreviation = title.length > 20 ? title.slice(0, 18) + ".." : title;
      updateStep(3, { status: "done", message: abbreviation });
    }

    // Step 5: Summarize (quick single query, also indexes PDF for QA)
    logIngest("Summarizing paper (may take ~30s)...");
    updateStep(4, { status: "running", message: "May take a minute..." });
    const summarizeRes = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pdf_path: pdfPath, arxiv_id: arxivId }),
    });
    if (!summarizeRes.ok) {
      const errText = await summarizeRes.text();
      logIngest(`Error: HTTP ${summarizeRes.status}`);
      try {
        const errJson = JSON.parse(errText);
        logIngest(`Details: ${errJson.detail || errText}`);
      } catch {
        logIngest(`Details: ${errText.slice(0, 200)}`);
      }
      updateStep(4, { status: "error", message: `HTTP ${summarizeRes.status}` });
      setIsIngesting(false);
      return;
    }
    const summarizeData = await summarizeRes.json();
    summary = summarizeData.summary;
    logIngest("Summary generated successfully");
    updateStep(4, { status: "done", message: "Done" });

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
        abbreviation,
      }),
    });
    if (!saveRes.ok) {
      updateStep(5, { status: "error", message: `HTTP ${saveRes.status}` });
    } else {
      updateStep(5, { status: "done", message: "Saved" });
    }

    // Refresh tree after a short delay — gives place_paper_in_tree time to complete
    setTimeout(async () => {
      const treeRes = await fetch("/api/tree");
      if (treeRes.ok) {
        const treeData = await treeRes.json();
        setTaxonomy(treeData);
        logIngest("Tree refreshed — paper placed in category");
      }
    }, 1500);

    // Prefetch auxiliary data in background (don't wait)
    fetch("/api/papers/prefetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arxiv_id: arxivId, title }),
    }).catch(() => {}); // Ignore errors, this is optional

      setUnifiedIngestInput("");
      setIsIngesting(false);
    }
  };

  const handleSlackIngest = async () => {
    if (!slackToken.trim()) {
      logIngest("Error: Slack token is required");
      return;
    }

    setIsBatchIngesting(true);
    setBatchResults(null);
    clearIngestLog();
    setShowSlackCredentialPrompt(false);
    logIngest(`Starting Slack channel ingest from: ${pendingSlackChannel}`);

    try {
      const res = await fetch("/api/papers/batch-ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          slack_channel: pendingSlackChannel,
          slack_token: slackToken,
          skip_existing: true,
        }),
      });

      if (!res.ok) {
        const errText = await res.text();
        logIngest(`Error: HTTP ${res.status}`);
        try {
          const errJson = JSON.parse(errText);
          // Display progress_log if available (for Slack ingestion errors)
          if (errJson.detail?.progress_log && Array.isArray(errJson.detail.progress_log)) {
            errJson.detail.progress_log.forEach((logMessage: string) => {
              logIngest(logMessage);
            });
          }
          // Display error message
          const errorMsg = errJson.detail?.message || errJson.detail || errText;
          logIngest(`Details: ${errorMsg}`);
        } catch {
          logIngest(`Details: ${errText.slice(0, 200)}`);
        }
        setIsBatchIngesting(false);
        return;
      }

      const data = await res.json();
      
      // Display progress log if available (for Slack ingestion)
      if (data.progress_log && Array.isArray(data.progress_log)) {
        data.progress_log.forEach((logMessage: string) => {
          logIngest(logMessage);
        });
      }
      
      setBatchResults(data);
      logIngest(`Completed: ${data.success} success, ${data.skipped} skipped, ${data.errors} errors`);

      // Refresh tree
      const treeRes = await fetch("/api/tree");
      if (treeRes.ok) {
        const treeData = await treeRes.json();
        setTaxonomy(treeData);
        logIngest("Tree refreshed");
      }

      // Clear token and pending channel
      setSlackToken("");
      setPendingSlackChannel("");
    } catch (err) {
      logIngest(`Error: ${err}`);
    } finally {
      setIsBatchIngesting(false);
      logIngest("Done");
    }
  };

  // Convert taxonomy to tree-compatible format

  // Find original node by node_id (or name for root) for details display
  const findNode = useCallback((tree: PaperNode, nodeId: string): PaperNode | null => {
    // Match by node_id if available, otherwise by name (for root node)
    if (tree.node_id === nodeId || (!tree.node_id && tree.name === nodeId)) {
      return tree;
    }
    for (const child of tree.children || []) {
      const found = findNode(child, nodeId);
      if (found) return found;
    }
    return null;
  }, []);

  // Get direct child categories of a node
  const getChildCategories = useCallback((node: PaperNode): PaperNode[] => {
    if (!node.children) return [];
    return node.children.filter((c) => c.node_type === "category" || (!c.paper_id && !c.attributes?.arxivId));
  }, []);

  // Count all papers under a node (including all descendants)
  const countAllPapers = useCallback((node: PaperNode): number => {
    if (!node.children) return 0;
    let count = 0;
    for (const child of node.children) {
      if (child.node_type === "paper" || child.paper_id || child.attributes?.arxivId) {
        count += 1;
      } else {
        count += countAllPapers(child);
      }
    }
    return count;
  }, []);

  // Check if a node is a category (not a paper)
  const isCategory = useCallback((node: PaperNode): boolean => {
    return node.node_type === "category" || (!node.paper_id && !node.attributes?.arxivId);
  }, []);
  
  // Search function - supports both paper and category modes
  const performSearch = useCallback((query: string, mode: "paper" | "category" = searchMode as "paper" | "category") => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }
    
    const results: PaperNode[] = [];
    const searchLower = query.toLowerCase();
    
    const normalizeArxiv = (value?: string | null) =>
      (value || "").toLowerCase().replace(/v\d+$/i, "");

    const searchPapers = (node: PaperNode) => {
      // Search in paper nodes (have arxivId)
      if (node.attributes?.arxivId) {
        const titleMatch = node.attributes.title?.toLowerCase().includes(searchLower);
        const nameMatch = node.name.toLowerCase().includes(searchLower);
        const authorMatch = node.attributes.authors?.some(author => 
          author.toLowerCase().includes(searchLower)
        );
        const arxivIdMatch = normalizeArxiv(node.attributes.arxivId).includes(searchLower);
        const arxivWithVersionMatch = node.attributes.arxivId.toLowerCase().includes(searchLower);
        
        if (titleMatch || nameMatch || authorMatch || arxivIdMatch || arxivWithVersionMatch) {
          results.push(node);
        }
      }
      
      // Recursively search children
      if (node.children) {
        node.children.forEach(child => searchPapers(child));
      }
    };

    const searchCategories = (node: PaperNode) => {
      // Search in category nodes (don't have arxivId)
      if (!node.attributes?.arxivId) {
        const nameMatch = node.name.toLowerCase().includes(searchLower);
        const nodeIdMatch = (node.node_id || "").toLowerCase().includes(searchLower);
        
        if (nameMatch || nodeIdMatch) {
          results.push(node);
        }
      }
      
      // Recursively search children
      if (node.children) {
        node.children.forEach(child => searchCategories(child));
      }
    };
    
    if (mode === "paper") {
      // Search papers
      if (taxonomy.children) {
        taxonomy.children.forEach(category => {
          if (category.children) {
            category.children.forEach(paper => searchPapers(paper));
          }
        });
      }
    } else if (mode === "category") {
      // Search categories - include root children and all nested categories
      if (taxonomy.children) {
        taxonomy.children.forEach(category => searchCategories(category));
      }
    }
    
    setSearchResults(results.slice(0, 10)); // Limit to 10 results
  }, [taxonomy, searchMode]);

  const handleNodeClick = useCallback(async (nodeId: string) => {
    const node = findNode(taxonomy, nodeId);
    setSelectedNode(node);
    setActivePanel("explorer");
    setStructuredAnalysis(null);
    
    // Reset states
    setRepos([]);
    setReferences([]);
    setSimilarPapers([]);
    setQueryHistory([]);
    setRefExplanations({});
    
    // Load cached data if this is a paper node
    if (node?.attributes?.arxivId) {
      setIsLoadingCachedData(true);
      try {
        const res = await fetch("/api/papers/cached-data", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ arxiv_id: node.attributes.arxivId }),
        });
        if (res.ok) {
          const data = await res.json();
          // Restore cached data
          if (data.repos?.length > 0) setRepos(data.repos);
          if (data.references?.length > 0) {
            setReferences(data.references);
            // Pre-populate cached explanations
            const cached: Record<number, string> = {};
            for (const ref of data.references) {
              if (ref.explanation) cached[ref.id] = ref.explanation;
            }
            setRefExplanations(cached);
          }
          if (data.similar_papers?.length > 0) setSimilarPapers(data.similar_papers);
          if (data.queries?.length > 0) setQueryHistory(data.queries);
          // Restore structured analysis if available
          if (data.structured_summary) {
            setStructuredAnalysis(data.structured_summary);
          }
        }
      } catch (e) {
        console.error("Failed to load cached data:", e);
      } finally {
        setIsLoadingCachedData(false);
      }
    }
  }, [taxonomy, findNode]);

  const handleNodeRightClick = useCallback((event: React.MouseEvent, nodeId: string) => {
    event.preventDefault();
    event.stopPropagation();
    const node = findNode(taxonomy, nodeId);
    if (node?.attributes?.arxivId) {
      setContextMenu({
        visible: true,
        x: event.clientX,
        y: event.clientY,
        node,
      });
    }
  }, [taxonomy, findNode]);

  const handleRemoveNode = async (node: PaperNode) => {
    if (!node.attributes?.arxivId) return;
    
    const arxivId = node.attributes.arxivId;
    const confirmed = window.confirm(
      `Are you sure you want to remove "${node.name}"?\n\nThis will permanently delete:\n- The paper from the database\n- All associated references, queries, and similar papers\n- The tree node\n\nThis action cannot be undone.`
    );
    
    if (!confirmed) return;
    
    clearFeatureLog();
    logFeature(`Removing paper: ${arxivId}...`);
    
    try {
      const res = await fetch("/api/papers/delete", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id: arxivId }),
      });
      
      if (res.ok) {
        logFeature(`✓ Paper ${arxivId} removed successfully`);
        // Reload tree to reflect changes
        const treeRes = await fetch("/api/tree");
        if (treeRes.ok) {
          const treeData = await treeRes.json();
          setTaxonomy(treeData);
        }
        // Clear selection if we just deleted the selected node
        if (selectedNode?.attributes?.arxivId === arxivId) {
          setSelectedNode(null);
          setActivePanel("explorer");
        }
      } else {
        const errText = await res.text();
        logFeature(`✗ Error: ${res.status} - ${errText}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
    }
    logFeature("Done");
  };

  // Feature handlers
  const handleFindRepos = async (node: PaperNode) => {
    if (!node.attributes?.arxivId) {
      logFeature("Error: No arXiv ID found for this paper");
      return;
    }
    clearFeatureLog();
    setSelectedNode(node);
    setActivePanel("repos");
    setIsLoadingFeature(true);
    setRepos([]);
    
    const arxivId = node.attributes.arxivId;
    const title = node.attributes.title || node.name;
    
    logFeature(`Starting GitHub repo search for: ${arxivId}`);
    logFeature(`Paper title: ${title}`);
    logFeature("Step 1: Querying Papers With Code API...");
    
    try {
      const res = await fetch("/api/repos/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id: arxivId, title }),
      });
      
      if (res.ok) {
        const data = await res.json();
        const repos = data.repos || [];
        setRepos(repos);
        
        if (data.from_cache) {
          logFeature("Retrieved from cache");
        } else {
          logFeature("Step 2: Querying GitHub Search API...");
        }
        
        if (repos.length > 0) {
          logFeature(`✓ Found ${repos.length} repositories`);
          const official = repos.filter((r: RepoResult) => r.is_official);
          if (official.length > 0) {
            logFeature(`  - ${official.length} official repo(s)`);
          }
        } else {
          logFeature("✗ No repositories found");
        }
      } else {
        logFeature(`✗ Error: HTTP ${res.status}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
      console.error("Failed to fetch repos:", e);
    } finally {
      setIsLoadingFeature(false);
      logFeature("Done");
    }
  };

  const handleFetchReferences = async (node: PaperNode) => {
    if (!node.attributes?.arxivId) {
      logFeature("Error: No arXiv ID found for this paper");
      return;
    }
    clearFeatureLog();
    setSelectedNode(node);
    setActivePanel("references");
    setIsLoadingFeature(true);
    setReferences([]);
    setRefExplanations({});
    
    const arxivId = node.attributes.arxivId;
    logFeature(`Starting reference extraction for: ${arxivId}`);
    logFeature("Step 1: Querying Semantic Scholar API...");
    
    try {
      const res = await fetch("/api/references/fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id: arxivId }),
      });
      
      if (res.ok) {
        const data = await res.json();
        const refs = data.references || [];
        setReferences(refs);
        
        if (data.from_cache) {
          logFeature("Retrieved from cache");
        }
        
        // Pre-populate cached explanations
        const cached: Record<number, string> = {};
        for (const ref of refs) {
          if (ref.explanation) {
            cached[ref.id] = ref.explanation;
          }
        }
        setRefExplanations(cached);
        
        if (refs.length > 0) {
          logFeature(`✓ Found ${refs.length} references`);
          const withArxiv = refs.filter((r: Reference) => r.cited_arxiv_id);
          logFeature(`  - ${withArxiv.length} with arXiv IDs`);
        } else {
          logFeature("✗ No references found");
        }
      } else {
        logFeature(`✗ Error: HTTP ${res.status}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
      console.error("Failed to fetch references:", e);
    } finally {
      setIsLoadingFeature(false);
      logFeature("Hover over references for LLM explanations");
      logFeature("Done");
    }
  };

  const handleFindSimilar = async (node: PaperNode) => {
    if (!node.attributes?.arxivId) {
      logFeature("Error: No arXiv ID found for this paper");
      return;
    }
    clearFeatureLog();
    setSelectedNode(node);
    setActivePanel("similar");
    setIsLoadingFeature(true);
    setSimilarPapers([]);
    
    const arxivId = node.attributes.arxivId;
    logFeature(`Starting similar paper search for: ${arxivId}`);
    logFeature("Step 1: Querying Semantic Scholar Recommendations API...");
    logFeature("Searching 200M+ papers on the internet...");
    
    try {
      const res = await fetch("/api/papers/similar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arxiv_id: arxivId }),
      });
      
      if (res.ok) {
        const data = await res.json();
        const papers = data.similar_papers || [];
        setSimilarPapers(papers);
        
        if (data.from_cache) {
          logFeature("Retrieved from cache");
        }
        
        if (papers.length > 0) {
          logFeature(`✓ Found ${papers.length} similar papers`);
          const withArxiv = papers.filter((p: SimilarPaper) => p.arxiv_id);
          logFeature(`  - ${withArxiv.length} available on arXiv`);
        } else {
          logFeature("✗ No similar papers found");
        }
      } else {
        logFeature(`✗ Error: HTTP ${res.status}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
      console.error("Failed to find similar papers:", e);
    } finally {
      setIsLoadingFeature(false);
      logFeature("Done");
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
    // Auto-trigger ingestion
    setUnifiedIngestInput(paper.arxiv_id);
    setActivePanel("explorer");
    // Trigger ingestion after a short delay to allow state update
    setTimeout(() => {
      handleUnifiedIngest();
    }, 100);
  };

  const handleAddReference = async (ref: Reference) => {
    if (!ref.cited_arxiv_id) return;
    // Auto-trigger ingestion
    setUnifiedIngestInput(ref.cited_arxiv_id);
    setActivePanel("explorer");
    // Trigger ingestion after a short delay to allow state update
    setTimeout(() => {
      handleUnifiedIngest();
    }, 100);
  };

  // Re-abbreviate handler for papers and categories
  const handleReabbreviate = async (node: PaperNode) => {
    // Show prompt dialog
    const result = window.prompt(
      `Re-abbreviate "${node.name}"\n\nEnter a custom name, or leave empty and click OK to auto-generate with AI.\nClick Cancel to abort.`
    );
    
    // If user clicked Cancel (result is null), do nothing
    if (result === null) {
      return;
    }
    
    const customName = result.trim() || undefined;
    
    try {
      let endpoint: string;
      let body: Record<string, string | undefined>;
      
      if (isCategory(node)) {
        // Category rename
        endpoint = "/api/categories/rename";
        body = { 
          node_id: node.node_id || node.name,
          custom_name: customName,
        };
      } else {
        // Paper re-abbreviation
        const arxivId = node.attributes?.arxivId;
        if (!arxivId) {
          alert("Cannot re-abbreviate: Paper has no arXiv ID");
          return;
        }
        endpoint = "/api/papers/reabbreviate";
        body = {
          arxiv_id: arxivId,
          custom_name: customName,
        };
      }
      
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      
      if (res.ok) {
        const data = await res.json();
        const newName = data.name || data.abbreviation;
        
        // Refresh tree to get updated node names
        const treeRes = await fetch("/api/tree");
        if (treeRes.ok) {
          const treeData = await treeRes.json();
          setTaxonomy(treeData);
        }
        
        // Update selected node locally for immediate feedback
        if (selectedNode && (selectedNode.node_id === node.node_id || selectedNode.name === node.name)) {
          setSelectedNode({ ...selectedNode, name: newName });
        }
        
        console.log(`Re-abbreviated to: ${newName}`);
      } else {
        const errorData = await res.json().catch(() => ({}));
        alert(`Failed to re-abbreviate: ${errorData.detail || res.statusText}`);
      }
    } catch (e) {
      console.error("Re-abbreviate error:", e);
      alert(`Error: ${e}`);
    }
  };

  const handleQuery = async () => {
    if (!selectedNode?.attributes?.arxivId || !queryInput.trim()) return;
    setIsQuerying(true);
    setQueryAnswer("");
    clearFeatureLog();
    const question = queryInput.trim();
    logFeature(`Querying: "${question}"`);
    logFeature("Searching paper for answer...");
    
    try {
      const res = await fetch("/api/qa", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          arxiv_id: selectedNode.attributes.arxivId,
          pdf_path: selectedNode.attributes.pdfPath || `storage/downloads/${selectedNode.attributes.arxivId}.pdf`,
          question: question,
          context: "",
        }),
      });
      if (res.ok) {
        const data = await res.json();
        const answer = data.answer || "No answer found.";
        setQueryAnswer(answer);
        logFeature(`✓ Answer generated${data.used_cache ? " (cached index)" : ""}`);
        
        // Add to query history (backend already persisted it)
        setQueryHistory(prev => [{
          id: Date.now(), // Temporary ID, actual ID is in DB
          question: question,
          answer: answer,
          created_at: new Date().toISOString(),
        }, ...prev]);
        
        // Clear input for next question
        setQueryInput("");
      } else {
        const errText = await res.text();
        logFeature(`✗ Error: ${res.status}`);
        setQueryAnswer(`Error: ${errText}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
      setQueryAnswer(`Error: ${e}`);
    } finally {
      setIsQuerying(false);
      logFeature("Done");
    }
  };

  const toggleQuerySelection = (queryId: number) => {
    setSelectedQueryIds(prev => {
      const newSet = new Set(prev);
      if (newSet.has(queryId)) {
        newSet.delete(queryId);
      } else {
        newSet.add(queryId);
      }
      return newSet;
    });
  };

  const handleMergeQueries = async () => {
    if (!selectedNode?.attributes?.arxivId || selectedQueryIds.size === 0) return;
    setIsMergingQueries(true);
    clearFeatureLog();
    logFeature(`Merging ${selectedQueryIds.size} Q&A pair(s) into summary...`);
    
    // Get selected Q&A pairs
    const selectedQa = queryHistory
      .filter(q => selectedQueryIds.has(q.id))
      .map(q => ({ question: q.question, answer: q.answer }));
    
    try {
      const res = await fetch("/api/summary/merge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          arxiv_id: selectedNode.attributes.arxivId,
          selected_qa: selectedQa,
        }),
      });
      
      if (res.ok) {
        const data = await res.json();
        logFeature(`✓ Merged ${data.merged_count} Q&A pair(s) into summary`);
        // Update the selected node's summary
        if (selectedNode) {
          setSelectedNode({
            ...selectedNode,
            attributes: {
              ...selectedNode.attributes,
              summary: data.summary,
            }
          });
        }
        // Clear selection
        setSelectedQueryIds(new Set());
        // Switch to explorer panel to show updated summary
        setActivePanel("explorer");
      } else {
        const errText = await res.text();
        logFeature(`✗ Error: ${res.status} - ${errText}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
    } finally {
      setIsMergingQueries(false);
      logFeature("Done");
    }
  };

  const handleDedupSummary = async () => {
    if (!selectedNode?.attributes?.arxivId) return;
    setIsDedupingSummary(true);
    clearFeatureLog();
    logFeature("Deduplicating summary...");
    
    try {
      const res = await fetch("/api/summary/dedup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          arxiv_id: selectedNode.attributes.arxivId,
        }),
      });
      
      if (res.ok) {
        const data = await res.json();
        const reduction = data.original_length - data.new_length;
        const percent = Math.round((reduction / data.original_length) * 100);
        logFeature(`✓ Removed duplicates (${reduction > 0 ? `-${percent}%` : "no changes"})`);
        // Update the selected node's summary
        if (selectedNode) {
          setSelectedNode({
            ...selectedNode,
            attributes: {
              ...selectedNode.attributes,
              summary: data.summary,
            }
          });
        }
      } else {
        const errText = await res.text();
        logFeature(`✗ Error: ${res.status} - ${errText}`);
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
    } finally {
      setIsDedupingSummary(false);
      logFeature("Done");
    }
  };

  const handleStructuredAnalysis = async () => {
    if (!selectedNode?.attributes?.arxivId) return;
    setIsLoadingStructured(true);
    setStructuredAnalysis(null);
    clearFeatureLog();
    logFeature(`Running detailed analysis for ${selectedNode.attributes.arxivId}...`);
    logFeature("This may take several minutes...");
    
    try {
      const res = await fetch("/api/qa/structured", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          arxiv_id: selectedNode.attributes.arxivId,
          pdf_path: selectedNode.attributes.pdfPath,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setStructuredAnalysis(data);
        logFeature(`✓ Analysis complete: ${data.components.length} components`);
      } else {
        const errText = await res.text();
        logFeature(`✗ Error: ${res.status}`);
        try {
          const errJson = JSON.parse(errText);
          logFeature(`Details: ${errJson.detail || errText}`);
        } catch {
          logFeature(`Details: ${errText.slice(0, 200)}`);
        }
      }
    } catch (e) {
      logFeature(`✗ Error: ${e}`);
    } finally {
      setIsLoadingStructured(false);
      logFeature("Done");
    }
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

  // Handle window resize for responsive design
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Determine responsive layout
  const isMobile = windowWidth < 768;
  const isTablet = windowWidth >= 768 && windowWidth < 1024;
  const isDesktop = windowWidth >= 1024;
  
  // Calculate panel widths based on responsive state and fullscreen mode
  const getPanelStyles = () => {
    if (isFullscreen === "tree") {
      return {
        left: { flex: "1 1 100%", display: "flex" },
        right: { flex: "0 0 0", display: "none" },
      };
    }
    if (isFullscreen === "details") {
      return {
        left: { flex: "0 0 0", display: "none" },
        right: { flex: "1 1 100%", display: "flex" },
      };
    }
    if (isMobile) {
      // Mobile: stack vertically, show one at a time
      return {
        left: { 
          flex: isFullscreen === null ? "1 1 50%" : "0 0 0", 
          display: isFullscreen === "details" ? "none" : "flex",
          minHeight: isFullscreen === null ? "50vh" : "100vh",
        },
        right: { 
          flex: isFullscreen === null ? "1 1 50%" : "1 1 100%", 
          display: "flex",
          minHeight: isFullscreen === null ? "50vh" : "100vh",
        },
      };
    }
    // Desktop/Tablet: side by side with resizable panels
    // Note: resizer takes ~20px, so we use flex: 1 for left when collapsed
    if (isRightCollapsed) {
      return {
        left: { flex: "1 1 auto", display: "flex" },
        right: { flex: "0 0 0", display: "none" },
      };
    }
    const rightPercent = rightPanelWidth;
    const leftPercent = 100 - rightPercent;
    return {
      left: { flex: `0 0 calc(${leftPercent}% - 10px)`, display: "flex" },
      right: { flex: `0 0 calc(${rightPercent}% - 10px)`, display: "flex" },
    };
  };
  
  const panelStyles = getPanelStyles();
  const toggleRightPanel = () => {
    if (isRightCollapsed) {
      setRightPanelWidth(50);
      setIsRightCollapsed(false);
    } else {
      setIsRightCollapsed(true);
    }
  };

  const treeLayout: TreeLayout | null = useMemo(() => {
    if (!taxonomy) return null;

    const wrapText = (text: string, maxChars: number) => {
      const words = text.split(" ");
      const lines: string[] = [];
      let current = "";
      for (const word of words) {
        if (!current) {
          current = word;
          continue;
        }
        if ((current + " " + word).length <= maxChars) {
          current += " " + word;
        } else {
          lines.push(current);
          current = word;
        }
      }
      if (current) lines.push(current);
      return lines.length ? lines : [text];
    };

    const dimsById: Record<string, { width: number; height: number; lineHeight: number; paddingY: number }> = {};
    const linesById: Record<string, string[]> = {};
    let maxCategoryHeight = 0;
    let maxPaperHeight = 0;
    let maxPaperColumnHeight = 0;

    const getNodeId = (node: PaperNode) => node.node_id || node.name;

    const isPaperNode = (node: PaperNode) =>
      node.node_type === "paper" ||
      !!node.paper_id ||
      (!!node.attributes?.arxivId && !(node.children && node.children.length));

    const buildDims = (node: PaperNode) => {
      const isPaper = isPaperNode(node);
      const nodeId = getNodeId(node);
      const maxChars = isPaper ? 22 : 20;
      const lineHeight = isPaper ? 16 : 18;
      const paddingY = isPaper ? 14 : 16;
      const width = isPaper ? 200 : 240;
      const lines = wrapText(node.name || "", maxChars);
      const height = Math.max(isPaper ? 44 : 52, lines.length * lineHeight + paddingY * 2);
      dimsById[nodeId] = { width, height, lineHeight, paddingY };
      linesById[nodeId] = lines;
      if (isPaper) {
        maxPaperHeight = Math.max(maxPaperHeight, height);
      } else {
        maxCategoryHeight = Math.max(maxCategoryHeight, height);
      }
    };

    const countPaperColumn = (node: PaperNode) => {
      const children = node.children || [];
      if (!children.length) return 0;
      if (children.every((c) => isPaperNode(c))) {
        const gap = 18;
        return children.length * (maxPaperHeight + gap);
      }
      return 0;
    };

    const collect = (node: PaperNode) => {
      buildDims(node);
      for (const child of node.children || []) {
        collect(child);
      }
    };

    collect(taxonomy);

    const collectPaperColumnHeights = (node: PaperNode) => {
      const columnHeight = countPaperColumn(node);
      maxPaperColumnHeight = Math.max(maxPaperColumnHeight, columnHeight);
      for (const child of node.children || []) {
        if (child.node_type === "category") {
          collectPaperColumnHeights(child);
        }
      }
    };

    collectPaperColumnHeights(taxonomy);

    const buildCategoryTree = (node: PaperNode): PaperNode => {
      const children = (node.children || []).filter((c) => !isPaperNode(c));
      return {
        ...node,
        children: children.map(buildCategoryTree),
      };
    };

    // Build lookup map from node_id to original taxonomy node (with paper children)
    const originalNodeById: Record<string, PaperNode> = {};
    const buildLookup = (node: PaperNode) => {
      const nodeId = getNodeId(node);
      originalNodeById[nodeId] = node;
      for (const child of node.children || []) {
        buildLookup(child);
      }
    };
    buildLookup(taxonomy);

    const categoryRoot = buildCategoryTree(taxonomy);
    const root = hierarchy(categoryRoot, (d) => d.children ?? []);

    const depthGap = maxCategoryHeight + Math.max(80, maxPaperColumnHeight) + 80;
    const layout = d3tree<PaperNode>()
      .nodeSize([260, depthGap])
      .separation((a, b) => (a.parent === b.parent ? 1.2 : 1.6));

    const laidOut = layout(root);
    const categoryNodes = laidOut.descendants();
    const categoryLinks = laidOut.links();

    const nodes: TreeLayout["nodes"] = [];
    const links: TreeLayout["links"] = [];

    for (const node of categoryNodes) {
      const original = node.data;
      const nodeId = getNodeId(original);
      nodes.push({
        x: node.x,
        y: node.y,
        data: original,
        nodeId,
        isPaper: false,
      });
    }

    for (const link of categoryLinks) {
      links.push({
        source: link.source,
        target: link.target,
        isPaper: false,
      });
    }

    const paperGap = 18;
    for (const catNode of categoryNodes) {
      const catId = getNodeId(catNode.data);
      // Use original taxonomy node (has paper children), not filtered catNode.data
      const originalNode = originalNodeById[catId];
      if (!originalNode) continue;
      const children = originalNode.children || [];
      if (!children.length) continue;
      const paperChildren = children.filter((c) => isPaperNode(c));
      if (!paperChildren.length) continue;
      const catDims = dimsById[catId];
      const startY = catNode.y + catDims.height / 2 + paperGap + maxPaperHeight / 2;
      paperChildren.forEach((paper, idx) => {
        const paperId = getNodeId(paper);
        const y = startY + idx * (maxPaperHeight + paperGap);
        nodes.push({
          x: catNode.x,
          y,
          data: paper,
          nodeId: paperId,
          isPaper: true,
        });
        links.push({
          source: { x: catNode.x, y: catNode.y, data: originalNode },
          target: { x: catNode.x, y, data: paper },
          isPaper: true,
        });
      });
    }

    if (!nodes.length) return null;
    const padding = 60;
    const minX = Math.min(...nodes.map((n) => n.x - dimsById[n.nodeId].width / 2));
    const maxX = Math.max(...nodes.map((n) => n.x + dimsById[n.nodeId].width / 2));
    const minY = Math.min(...nodes.map((n) => n.y - dimsById[n.nodeId].height / 2));
    const maxY = Math.max(...nodes.map((n) => n.y + dimsById[n.nodeId].height / 2));

    return {
      nodes,
      links,
      width: maxX - minX + padding * 2,
      height: maxY - minY + padding * 2,
      offsetX: padding - minX,
      offsetY: padding - minY,
      linesById,
      dimsById,
    };
  }, [taxonomy]);

  // Auto-zoom to fit tree on initial load
  useEffect(() => {
    if (!treeLayout || hasAutoZoomed || !treeContainerRef.current) return;
    
    const container = treeContainerRef.current;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    
    if (containerWidth === 0 || containerHeight === 0) return;
    
    // Calculate zoom to fit tree in container with 10% margin
    const marginFactor = 0.9;
    const scaleX = (containerWidth * marginFactor) / treeLayout.width;
    const scaleY = (containerHeight * marginFactor) / treeLayout.height;
    const fitZoom = Math.min(scaleX, scaleY, 1); // Don't zoom in beyond 100%
    
    setZoomLevel(Math.max(0.25, fitZoom)); // Ensure minimum 25%
    setHasAutoZoomed(true);
  }, [treeLayout, hasAutoZoomed]);

  return (
    <TooltipProvider>
      <div style={{ display: "flex", height: "100vh", flexDirection: isMobile && !isFullscreen ? "column" : "row", overflow: "hidden" }}>
      {/* Left panel: Tree visualization */}
      <div style={{ ...panelStyles.left, borderRight: isMobile ? "none" : "1px solid #e5e5e5", borderBottom: isMobile && !isFullscreen ? "1px solid #e5e5e5" : "none", display: "flex", flexDirection: "column", position: "relative", overflow: "hidden", minWidth: 0 }}>
        <div style={{ padding: isMobile ? "0.75rem" : "1rem", borderBottom: "1px solid #e5e5e5", backgroundColor: "#fafafa", fontSize: `${panelFontSize}px` }}>
          {/* Title Row - Centered with Loading Indicator and Settings */}
          <div className="flex justify-center items-center gap-2 mb-3">
            <h1 style={{ margin: 0, fontSize: isMobile ? "1.5rem" : "1.75rem", fontWeight: 600 }}>Paper Curator</h1>
            {isLoadingTree && (
              <div className="flex items-center gap-1 text-sm text-gray-500">
                <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Loading tree...</span>
              </div>
            )}
            <button
              onClick={handleOpenSettings}
              className="p-1.5 rounded hover:bg-gray-200 transition-colors"
              title="Settings"
            >
              <Settings className="h-5 w-5 text-gray-600" />
            </button>
          </div>
          {/* Controls Row: Ingest + Reclassify + Search */}
          <div className="flex items-center gap-2">
            {/* Ingest Input */}
            <input
              type="text"
              value={unifiedIngestInput}
              onChange={(e) => setUnifiedIngestInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !isIngesting && !isBatchIngesting && unifiedIngestInput.trim()) {
                  handleUnifiedIngest();
                }
              }}
              placeholder="arXiv URL/ID or path..."
              disabled={isIngesting || isBatchIngesting}
              className="flex-1 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100"
            />
            <button
              onClick={handleUnifiedIngest}
              disabled={isIngesting || isBatchIngesting || !unifiedIngestInput.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded font-medium disabled:bg-gray-400 hover:bg-blue-700 whitespace-nowrap"
            >
              {(isIngesting || isBatchIngesting) ? "Processing..." : "Ingest"}
            </button>
            <button
              onClick={handleCategorizePartial}
              disabled={isCategorizing}
              className="px-4 py-2 bg-purple-600 text-white rounded font-medium disabled:bg-gray-400 hover:bg-purple-700 whitespace-nowrap"
              title="Re-cluster dirty nodes (fast)"
            >
              {isCategorizing ? "Categorizing..." : "Categorize"}
            </button>
            <button
              onClick={handleCategorizeFull}
              disabled={isCategorizing}
              className="px-4 py-2 bg-purple-800 text-white rounded font-medium disabled:bg-gray-400 hover:bg-purple-900 whitespace-nowrap"
              title="Full rebuild from scratch (slow, 30+ min)"
            >
              Full Rebuild
            </button>
            {/* Search with mode dropdown */}
            <div className="relative flex-1 flex">
              {/* Search mode dropdown */}
              <select
                value={searchMode}
                onChange={(e) => setSearchMode(e.target.value as "paper" | "category" | "topic")}
                className="px-2 py-2 border border-r-0 border-gray-300 rounded-l bg-gray-50 text-sm text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500"
                title="Search mode"
              >
                <option value="paper">Paper</option>
                <option value="category">Category</option>
                <option value="topic">Topic</option>
              </select>
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    if (searchMode !== "topic") {
                      performSearch(e.target.value);
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && searchMode === "topic" && searchQuery.trim()) {
                      e.preventDefault();
                      handleTopicSearch(searchQuery.trim());
                    }
                  }}
                  onFocus={() => setIsSearchFocused(true)}
                  onBlur={() => setTimeout(() => setIsSearchFocused(false), 200)}
                  placeholder={searchMode === "paper" ? "Search papers..." : searchMode === "category" ? "Search categories..." : "Enter topic to find papers..."}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-r focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              {isSearchFocused && searchResults.length > 0 && (
                <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto">
                  {searchResults.map((result, idx) => {
                    const isCategory = !result.attributes?.arxivId;
                    return (
                      <div
                        key={idx}
                        onClick={() => {
                          const nodeId = result.node_id || result.name;
                          handleNodeClick(nodeId);
                          setSearchQuery("");
                          setSearchResults([]);
                          // Center on the selected node after a short delay
                          setTimeout(() => centerOnNode(nodeId), 100);
                        }}
                        className="px-4 py-2 hover:bg-gray-100 cursor-pointer border-b border-gray-100 last:border-b-0"
                      >
                        <div className="flex items-center gap-2">
                          <span className={isCategory ? "text-purple-600" : "text-blue-600"}>
                            {isCategory ? "📁" : "📄"}
                          </span>
                          <div className="font-medium">{result.name}</div>
                        </div>
                        {isCategory ? (
                          <div className="text-gray-500 text-sm mt-1 ml-6">
                            {result.children?.length || 0} child nodes
                          </div>
                        ) : (
                          <>
                            {result.attributes?.title && (
                              <div className="text-gray-600 mt-1 ml-6">{result.attributes.title}</div>
                            )}
                            {result.attributes?.authors && result.attributes.authors.length > 0 && (
                              <div className="text-gray-500 mt-1 ml-6">
                                {result.attributes.authors.slice(0, 3).join(", ")}
                                {result.attributes.authors.length > 3 && ` +${result.attributes.authors.length - 3} more`}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
              </div>
            </div>
          </div>
        </div>
        {/* Tree Diagram Container - Outer wrapper for fixed overlays */}
        <div
          style={{
            flex: 1,
            position: "relative",
            overflow: "hidden",
            backgroundColor: "#f8fafc",
          }}
        >
          {/* Zoom Controls - Fixed position relative to container */}
          {treeLayout && (
            <div
              style={{
                position: "absolute",
                top: "10px",
                left: "10px",
                zIndex: 20,
                display: "flex",
                flexDirection: "column",
                gap: "4px",
                backgroundColor: "white",
                borderRadius: "8px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
                padding: "4px",
              }}
            >
              <button
                onClick={() => {
                  if (!treeContainerRef.current) {
                    setZoomLevel((z) => Math.min(3, z + 0.1));
                    return;
                  }
                  const container = treeContainerRef.current;
                  const centerX = container.clientWidth / 2;
                  const centerY = container.clientHeight / 2;
                  const scrollLeft = container.scrollLeft;
                  const scrollTop = container.scrollTop;
                  
                  setZoomLevel((prevZoom) => {
                    const newZoom = Math.min(3, prevZoom + 0.1);
                    const scale = newZoom / prevZoom;
                    
                    // Point at center in content coordinates
                    const pointX = scrollLeft + centerX;
                    const pointY = scrollTop + centerY;
                    
                    // Keep center stationary
                    container.scrollLeft = Math.max(0, pointX * scale - centerX);
                    container.scrollTop = Math.max(0, pointY * scale - centerY);
                    
                    return newZoom;
                  });
                }}
                style={{
                  width: "32px",
                  height: "32px",
                  border: "1px solid #e5e7eb",
                  borderRadius: "6px",
                  backgroundColor: "#fff",
                  cursor: "pointer",
                  fontSize: "18px",
                  fontWeight: "bold",
                }}
                title="Zoom in (Ctrl+Scroll)"
              >
                +
              </button>
              <button
                onClick={() => setZoomLevel(1)}
                style={{
                  width: "32px",
                  height: "32px",
                  border: "1px solid #e5e7eb",
                  borderRadius: "6px",
                  backgroundColor: "#fff",
                  cursor: "pointer",
                  fontSize: "11px",
                }}
                title="Reset zoom"
              >
                {Math.round(zoomLevel * 100)}%
              </button>
              <button
                onClick={() => {
                  if (!treeContainerRef.current) {
                    setZoomLevel((z) => Math.max(0.25, z - 0.1));
                    return;
                  }
                  const container = treeContainerRef.current;
                  const centerX = container.clientWidth / 2;
                  const centerY = container.clientHeight / 2;
                  const scrollLeft = container.scrollLeft;
                  const scrollTop = container.scrollTop;
                  
                  setZoomLevel((prevZoom) => {
                    const newZoom = Math.max(0.25, prevZoom - 0.1);
                    const scale = newZoom / prevZoom;
                    
                    // Point at center in content coordinates
                    const pointX = scrollLeft + centerX;
                    const pointY = scrollTop + centerY;
                    
                    // Keep center stationary
                    container.scrollLeft = Math.max(0, pointX * scale - centerX);
                    container.scrollTop = Math.max(0, pointY * scale - centerY);
                    
                    return newZoom;
                  });
                }}
                style={{
                  width: "32px",
                  height: "32px",
                  border: "1px solid #e5e7eb",
                  borderRadius: "6px",
                  backgroundColor: "#fff",
                  cursor: "pointer",
                  fontSize: "18px",
                  fontWeight: "bold",
                }}
                title="Zoom out"
              >
                −
              </button>
            </div>
          )}

          {/* Minimap - Fixed at bottom-left */}
          {treeLayout && (
            <div
              style={{
                position: "absolute",
                bottom: "10px",
                left: "10px",
                zIndex: 20,
                width: "150px",
                height: "100px",
                backgroundColor: "white",
                border: "1px solid #e5e7eb",
                borderRadius: "8px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
                overflow: "hidden",
              }}
            >
              <svg
                width="100%"
                height="100%"
                viewBox={`0 0 ${treeLayout.width} ${treeLayout.height}`}
                preserveAspectRatio="xMidYMid meet"
              >
                {/* Simplified tree outline for minimap */}
                <g transform={`translate(${treeLayout.offsetX}, ${treeLayout.offsetY})`}>
                  {treeLayout.links.map((link, idx) => (
                    <line
                      key={`mini-link-${idx}`}
                      x1={link.source.x}
                      y1={link.source.y}
                      x2={link.target.x}
                      y2={link.target.y}
                      stroke="#cbd5e1"
                      strokeWidth={2}
                    />
                  ))}
                  {treeLayout.nodes.map((node) => (
                    <circle
                      key={`mini-node-${node.nodeId}`}
                      cx={node.x}
                      cy={node.y}
                      r={node.isPaper ? 3 : 5}
                      fill={node.isPaper ? "#94a3b8" : "#3b82f6"}
                    />
                  ))}
                </g>
                {/* Viewport indicator */}
                {treeContainerRef.current && (
                  <rect
                    x={viewportPos.x / zoomLevel}
                    y={viewportPos.y / zoomLevel}
                    width={treeContainerRef.current.clientWidth / zoomLevel}
                    height={treeContainerRef.current.clientHeight / zoomLevel}
                    fill="rgba(59, 130, 246, 0.2)"
                    stroke="#3b82f6"
                    strokeWidth={3}
                  />
                )}
              </svg>
            </div>
          )}

          {/* Scrollable inner container for tree */}
          <div
            ref={treeContainerRef}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              overflow: "auto",
            }}
            onScroll={(e) => {
              const target = e.currentTarget;
              setViewportPos({ x: target.scrollLeft, y: target.scrollTop });
            }}
            onWheel={(e) => {
              // Prevent browser zoom when Ctrl/Cmd is pressed - capture the event fully
              if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                e.stopPropagation();
                
                const container = e.currentTarget;
                const rect = container.getBoundingClientRect();
                
                // Cursor position relative to container viewport
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;
                
                // Current scroll position
                const scrollLeft = container.scrollLeft;
                const scrollTop = container.scrollTop;
                
                setZoomLevel((prevZoom) => {
                  // Smaller delta for smoother zoom
                  const zoomDelta = -e.deltaY * 0.0008;
                  const newZoom = Math.max(0.25, Math.min(3, prevZoom + zoomDelta));
                  const scale = newZoom / prevZoom;
                  
                  // Point under cursor in content coordinates (before zoom)
                  const pointX = scrollLeft + mouseX;
                  const pointY = scrollTop + mouseY;
                  
                  // Where that point should be after zoom to keep cursor stationary
                  const newScrollLeft = pointX * scale - mouseX;
                  const newScrollTop = pointY * scale - mouseY;
                  
                  // Apply immediately for smoothness
                  container.scrollLeft = Math.max(0, newScrollLeft);
                  container.scrollTop = Math.max(0, newScrollTop);
                  
                  return newZoom;
                });
              }
            }}
          >
          {!treeLayout ? (
            <div
              style={{
                textAlign: "center",
                color: "#64748b",
                padding: "40px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
              }}
            >
              <div>
                <div style={{ fontSize: "48px", marginBottom: "16px" }}>🌳</div>
                <h2 style={{ fontSize: "24px", fontWeight: 600, marginBottom: "8px", color: "#334155" }}>
                  No tree data loaded yet
                </h2>
                <p style={{ fontSize: "14px", maxWidth: "420px", lineHeight: 1.6 }}>
                  Ingest papers to build the tree, then re-classify to update the visualization.
                </p>
              </div>
            </div>
          ) : (
            <svg
              width={treeLayout.width * zoomLevel}
              height={treeLayout.height * zoomLevel}
              style={{ display: "block" }}
            >
              <g transform={`scale(${zoomLevel}) translate(${treeLayout.offsetX}, ${treeLayout.offsetY})`}>
                {treeLayout.links.map((link, idx) => {
                  const sourceId = link.source.data.node_id || link.source.data.name;
                  const targetId = link.target.data.node_id || link.target.data.name;
                  const sourceDims = treeLayout.dimsById[sourceId];
                  const targetDims = treeLayout.dimsById[targetId];
                  const x1 = link.source.x;
                  const y1 = link.source.y + sourceDims.height / 2;
                  const x2 = link.target.x;
                  const y2 = link.target.y - targetDims.height / 2;
                  const midY = (y1 + y2) / 2;
                  const d = link.isPaper
                    ? `M ${x1},${y1} L ${x2},${y2}`
                    : `M ${x1},${y1} C ${x1},${midY} ${x2},${midY} ${x2},${y2}`;
                  return (
                    <path
                      key={`link-${idx}`}
                      d={d}
                      fill="none"
                      stroke="#94a3b8"
                      strokeWidth={1.5}
                    />
                  );
                })}
                {treeLayout.nodes.map((node) => {
                  const dims = treeLayout.dimsById[node.nodeId];
                  const x = node.x - dims.width / 2;
                  const y = node.y - dims.height / 2;
                  const isPaper = node.data.node_type === "paper";
                  const nodeId = node.nodeId;
                  const lines = treeLayout.linesById[nodeId] || [node.data.name];
                  const isPulsing = pulsingNodeId === nodeId;
                  return (
                    <g
                      key={nodeId}
                      data-node-id={nodeId}
                      transform={`translate(${x}, ${y})`}
                      onClick={() => handleNodeClick(nodeId)}
                      onContextMenu={(e) => handleNodeRightClick(e, nodeId)}
                      style={{ cursor: "pointer" }}
                    >
                      <rect
                        width={dims.width}
                        height={dims.height}
                        rx={10}
                        ry={10}
                        fill={isPaper ? "#e2e8f0" : "#1d4ed8"}
                        stroke={isPulsing ? "#f59e0b" : (isPaper ? "#cbd5f5" : "#1e3a8a")}
                        strokeWidth={isPulsing ? 3 : 1}
                        style={isPulsing ? { animation: "pulse 0.5s ease-in-out 3" } : undefined}
                      />
                      <text
                        x={dims.width / 2}
                        y={dims.paddingY}
                        textAnchor="middle"
                        dominantBaseline="hanging"
                        fontSize={isPaper ? (uiConfig?.tree_paper_font_size || 20) : (uiConfig?.tree_category_font_size || 20)}
                        fontWeight={isPaper ? 500 : 600}
                        fill={isPaper ? "#334155" : "#ffffff"}
                      >
                        {lines.map((line, idx) => (
                          <tspan key={idx} x={dims.width / 2} dy={idx === 0 ? 0 : dims.lineHeight}>
                            {line}
                          </tspan>
                        ))}
                      </text>
                    </g>
                  );
                })}
              </g>
            </svg>
          )}
          </div>
        </div>
      </div>

      {/* Context Menu - Only Remove Paper option remains */}
      {contextMenu.visible && contextMenu.node && (
        <div
          style={{
            position: "fixed",
            top: contextMenu.y,
            left: contextMenu.x,
            backgroundColor: "white",
            border: "1px solid #e5e5e5",
            borderRadius: "8px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            zIndex: 1000,
            minWidth: "200px",
            overflow: "hidden",
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div style={{ padding: "0.5rem 1rem", borderBottom: "1px solid #eee", backgroundColor: "#f9f9f9" }}>
            <strong style={{ fontSize: "0.75rem", color: "#666" }}>
              {contextMenu.node.attributes?.arxivId || contextMenu.node.name}
            </strong>
          </div>
          <div
            style={{ 
              padding: "0.75rem 1rem", 
              cursor: "pointer", 
              display: "flex", 
              alignItems: "center", 
              gap: "0.5rem",
              color: "#dc2626",
            }}
            onClick={() => { handleRemoveNode(contextMenu.node!); setContextMenu({ ...contextMenu, visible: false }); }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#fef2f2")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "white")}
          >
            <span>🗑️</span> Remove Paper
          </div>
        </div>
      )}

      {/* Slack Credential Prompt Modal */}
      {showSlackCredentialPrompt && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 2000,
          }}
          onClick={() => {
            setShowSlackCredentialPrompt(false);
            setPendingSlackChannel("");
            setSlackToken("");
          }}
        >
          <div
            style={{
              backgroundColor: "white",
              borderRadius: "8px",
              padding: "24px",
              maxWidth: "500px",
              width: "90%",
              boxShadow: "0 8px 24px rgba(0,0,0,0.2)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-lg font-semibold mb-4">Slack Channel Access</h2>
            <p className="text-sm text-gray-600 mb-4">
              Enter your Slack User OAuth Token to access the channel:
            </p>
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">
                Channel: <span className="font-mono">{pendingSlackChannel}</span>
              </label>
              <label className="block text-sm font-medium mb-2">
                Slack Token (xoxp-...)
              </label>
              <input
                type="password"
                value={slackToken}
                onChange={(e) => setSlackToken(e.target.value)}
                placeholder="xoxp-..."
                className="w-full px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && slackToken.trim()) {
                    handleSlackIngest();
                  }
                  if (e.key === "Escape") {
                    setShowSlackCredentialPrompt(false);
                    setPendingSlackChannel("");
                    setSlackToken("");
                  }
                }}
                autoFocus
              />
              <p className="text-gray-500 mt-2">
                Token is not persisted and will be cleared after use. Get your token from{" "}
                <a
                  href="https://api.slack.com/apps"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  api.slack.com/apps
                </a>
              </p>
            </div>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => {
                  setShowSlackCredentialPrompt(false);
                  setPendingSlackChannel("");
                  setSlackToken("");
                }}
                className="px-4 py-2 text-sm border border-gray-300 rounded hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSlackIngest}
                disabled={!slackToken.trim() || isBatchIngesting}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
              >
                {isBatchIngesting ? "Processing..." : "Ingest"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Resizer handle with arrow toggle for desktop/tablet */}
      {!isMobile && !isFullscreen && (
        <div
          style={{
            width: isRightCollapsed ? "20px" : "8px",
            backgroundColor: "#e5e5e5",
            cursor: isRightCollapsed ? "pointer" : "col-resize",
            position: "relative",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
          onMouseDown={(e) => {
            if (isRightCollapsed) return; // Don't resize when collapsed
            e.preventDefault();
            const startX = e.clientX;
            const startRightWidth = rightPanelWidth;
            
            const handleMouseMove = (moveEvent: MouseEvent) => {
              const deltaX = moveEvent.clientX - startX;
              const deltaPercent = (deltaX / windowWidth) * 100;
              const newRightWidth = Math.max(20, Math.min(80, startRightWidth - deltaPercent));
              setRightPanelWidth(newRightWidth);
            };
            
            const handleMouseUp = () => {
              document.removeEventListener("mousemove", handleMouseMove);
              document.removeEventListener("mouseup", handleMouseUp);
            };
            
            document.addEventListener("mousemove", handleMouseMove);
            document.addEventListener("mouseup", handleMouseUp);
          }}
        >
          {/* Arrow toggle button centered on resizer */}
          <button
            onClick={toggleRightPanel}
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              width: "20px",
              height: "40px",
              border: "1px solid #d1d5db",
              borderRadius: "4px",
              backgroundColor: "#ffffff",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "12px",
              color: "#6b7280",
              boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
              zIndex: 10,
            }}
            title={isRightCollapsed ? "Show panel" : "Hide panel"}
          >
            {isRightCollapsed ? "←" : "→"}
          </button>
        </div>
      )}
      
      {/* Right panel: Details and ingest */}
      <div style={{ ...panelStyles.right, padding: isMobile ? "1rem" : "1.5rem", display: "flex", flexDirection: "column", backgroundColor: "#f9fafb", overflowY: "auto", position: "relative", fontSize: `${panelFontSize}px` }}>
        {/* Static "Details" header at the top */}
        <h1 className="text-xl font-bold mb-4 text-gray-800 text-center">Details</h1>
        
        {/* Paper Details Section - Accordion */}
        <Card className="flex-1 flex flex-col">
          <Accordion type="single" collapsible defaultValue="details" className="w-full flex-1 flex flex-col">
            <AccordionItem value="details" className="border-0 flex-1 flex flex-col">
              <AccordionTrigger className="px-4 py-3 hover:no-underline">
                <h2 className="font-semibold m-0" style={{ fontSize: `${panelFontSize + 2}px` }}>
                  {selectedNode && isCategory(selectedNode) ? "📁" : "📄"} {selectedNode ? selectedNode.name : "Details"}
                </h2>
              </AccordionTrigger>
              <AccordionContent className="flex-1 flex flex-col overflow-hidden">
                <div className="flex-1 flex flex-col overflow-hidden">
                  {/* Panel tabs */}

                  {/* Dynamic panel content */}
                  <div style={{ flex: 1, padding: "0.75rem", overflowY: "auto" }}>
                    {/* Loading skeleton for cached data */}
                    {isLoadingCachedData && (
                      <div className="space-y-4">
                        <div className="animate-pulse">
                          <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                        </div>
                      </div>
                    )}
                    {selectedNode && !isLoadingCachedData && (
                  <Tabs 
                    value={activePanel === "references" ? "refs" : activePanel} 
                    onValueChange={(value) => {
                      if (value === "refs") {
                        setActivePanel("references");
                      } else {
                        setActivePanel(value as any);
                      }
                    }}
                    className="w-full"
                  >
                    {/* Tabs row with font size controls */}
                    <div className="flex items-center justify-between mb-2">
                      <TabsList className="grid grid-cols-6 h-10">
                        <TabsTrigger value="explorer" className="px-1 text-xs">Explorer</TabsTrigger>
                        <TabsTrigger value="repos" className="px-1 text-xs">Repos</TabsTrigger>
                        <TabsTrigger value="refs" className="px-1 text-xs">Refs</TabsTrigger>
                        <TabsTrigger value="similar" className="px-1 text-xs">Similar</TabsTrigger>
                        <TabsTrigger value="query" className="px-1 text-xs">Query</TabsTrigger>
                        <TabsTrigger value="topic-query" className="px-1 text-xs">Topic</TabsTrigger>
                      </TabsList>
                      <div className="flex items-center gap-1 ml-2">
                        <button
                          onClick={() => setPanelFontSize((s) => Math.max(14, s - 1))}
                          className="w-7 h-7 border border-gray-300 rounded hover:bg-gray-100"
                          title="Decrease font size"
                        >
                          A-
                        </button>
                        <span className="text-gray-500 w-8 text-center">{panelFontSize}</span>
                        <button
                          onClick={() => setPanelFontSize((s) => Math.min(25, s + 1))}
                          className="w-7 h-7 border border-gray-300 rounded hover:bg-gray-100"
                          title="Increase font size"
                        >
                          A+
                        </button>
                      </div>
                    </div>
                    
                    {/* Debug Log Section - Always visible, collapsible */}
                    <div className="mb-3 border border-gray-200 rounded-lg bg-gray-50">
                      <button
                        onClick={() => setIsDebugPanelOpen(!isDebugPanelOpen)}
                        className="w-full flex items-center justify-between px-3 py-2 font-semibold text-gray-700 hover:bg-gray-100 rounded-t-lg"
                      >
                        <span>🐛 Debug Logs ({featureLog.length + ingestLog.length} entries)</span>
                        <span>{isDebugPanelOpen ? "▼" : "▶"}</span>
                      </button>
                      {isDebugPanelOpen && (
                        <div className="px-3 pb-3 max-h-40 overflow-y-auto">
                          {featureLog.length === 0 && ingestLog.length === 0 ? (
                            <p className="text-gray-400 italic">No logs yet</p>
                          ) : (
                            <div
                              className="bg-gray-900 rounded p-2 font-mono max-h-32 overflow-y-auto"
                              style={{ fontSize: `${Math.max(12, panelFontSize - 6)}px` }}
                            >
                              {[...ingestLog, ...featureLog].slice(-50).map((log, i) => (
                                <div key={i} className={`mb-0.5 ${log.includes("✓") ? "text-green-400" : log.includes("✗") ? "text-red-400" : log.includes("Error") ? "text-red-400" : "text-gray-300"}`}>
                                  {log}
                                </div>
                              ))}
                            </div>
                          )}
                          {(featureLog.length > 0 || ingestLog.length > 0) && (
                            <div className="flex gap-2 mt-2">
                              <button onClick={clearFeatureLog} className="text-gray-500 hover:text-gray-700">Clear Feature</button>
                              <button onClick={clearIngestLog} className="text-gray-500 hover:text-gray-700">Clear Ingest</button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                    
                    {/* Explorer Panel - Combined with Details */}
                    <TabsContent value="explorer" className="mt-0" style={{ fontSize: `${panelFontSize}px` }}>
                      {/* Category Details */}
                      {selectedNode && isCategory(selectedNode) && (
                        <div className="mb-4 pb-4 border-b border-gray-200">
                          <div className="flex items-center justify-between mb-3">
                            <h2 className="font-semibold" style={{ fontSize: `${panelFontSize + 2}px` }}>Category Details</h2>
                            <button
                              onClick={() => handleReabbreviate(selectedNode)}
                              className="px-2 py-1 text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 rounded transition-colors"
                              title="Re-generate category name using AI or enter custom name"
                            >
                              Re-name
                            </button>
                          </div>
                          <h3 className="font-semibold mb-2">{selectedNode.name}</h3>
                          
                          {/* Ancestry chain (including current node) */}
                          {(() => {
                            const ancestors = getAncestorChain(selectedNode);
                            const fullChain = [...ancestors, selectedNode];
                            return (
                              <div className="mb-3 text-sm">
                                <strong className="text-gray-600">Ancestry:</strong>
                                <div className="mt-1">
                                  {fullChain.map((node, idx) => {
                                    const isCurrentNode = idx === fullChain.length - 1;
                                    return (
                                      <div
                                        key={node.node_id || idx}
                                        style={{ marginLeft: `${idx * 16}px` }}
                                        className="py-0.5"
                                      >
                                        {isCurrentNode ? (
                                          <span className="text-gray-800 font-semibold">
                                            {idx === 0 ? "📂" : "└─"} {node.name}
                                          </span>
                                        ) : (
                                          <button
                                            onClick={() => {
                                              setSelectedNode(node);
                                              const nodeId = node.node_id || node.name;
                                              setTimeout(() => centerOnNode(nodeId), 100);
                                            }}
                                            className="text-blue-600 hover:underline text-left"
                                          >
                                            {idx === 0 ? "📂" : "└─"} {node.name}
                                          </button>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            );
                          })()}
                          
                          {/* Child categories */}
                          {(() => {
                            const childCats = getChildCategories(selectedNode);
                            return childCats.length > 0 && (
                              <div className="mb-3">
                                <p className="text-gray-600 mb-1">
                                  <strong>{childCats.length}</strong> child {childCats.length === 1 ? "category" : "categories"}:
                                </p>
                                <div className="pl-2 text-gray-700">
                                  {childCats.map((cat, idx) => (
                                    <span key={cat.node_id || idx}>
                                      <button
                                        onClick={() => { setSelectedNode(cat); const catId = cat.node_id || cat.name; setTimeout(() => centerOnNode(catId), 100); }}
                                        className="text-blue-600 hover:underline"
                                      >
                                        {cat.name}
                                      </button>
                                      {idx < childCats.length - 1 && ", "}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            );
                          })()}
                          
                          {/* Total papers */}
                          <p className="text-gray-600">
                            <strong>{countAllPapers(selectedNode)}</strong> papers in this category (including subcategories)
                          </p>
                        </div>
                      )}
                      
                      {/* Paper Details */}
                      {selectedNode && !isCategory(selectedNode) && selectedNode.attributes && (
                        <div className="mb-4 pb-4 border-b border-gray-200">
                          <div className="flex items-center justify-between mb-3">
                            <h2 className="font-semibold" style={{ fontSize: `${panelFontSize + 2}px` }}>Paper Details</h2>
                            <button
                              onClick={() => handleReabbreviate(selectedNode)}
                              className="px-2 py-1 text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 rounded transition-colors"
                              title="Re-generate paper abbreviation using AI or enter custom name"
                            >
                              Re-abbreviate
                            </button>
                          </div>
                          <h3 className="font-semibold mb-2">{selectedNode.attributes.title || selectedNode.name}</h3>
                          
                          {/* Ancestry chain (including current node) */}
                          {(() => {
                            const ancestors = getAncestorChain(selectedNode);
                            const fullChain = [...ancestors, selectedNode];
                            return (
                              <div className="mb-3 text-sm">
                                <strong className="text-gray-600">Ancestry:</strong>
                                <div className="mt-1">
                                  {fullChain.map((node, idx) => {
                                    const isCurrentNode = idx === fullChain.length - 1;
                                    return (
                                      <div
                                        key={node.node_id || idx}
                                        style={{ marginLeft: `${idx * 16}px` }}
                                        className="py-0.5"
                                      >
                                        {isCurrentNode ? (
                                          <span className="text-gray-800 font-semibold">
                                            {idx === 0 ? "📂" : "└─"} {node.name}
                                          </span>
                                        ) : (
                                          <button
                                            onClick={() => {
                                              setSelectedNode(node);
                                              const nodeId = node.node_id || node.name;
                                              setTimeout(() => centerOnNode(nodeId), 100);
                                            }}
                                            className="text-blue-600 hover:underline text-left"
                                          >
                                            {idx === 0 ? "📂" : "└─"} {node.name}
                                          </button>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            );
                          })()}
                          
                          {selectedNode.attributes.arxivId && (
                            <p className="text-gray-600 mb-2">
                              <strong>arXiv:</strong>{" "}
                              <a href={`https://arxiv.org/abs/${selectedNode.attributes.arxivId}`} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                                {selectedNode.attributes.arxivId}
                              </a>
                            </p>
                          )}
                          {selectedNode.attributes.authors && selectedNode.attributes.authors.length > 0 && (
                            <p className="text-gray-600 mb-2">
                              <strong>Authors:</strong> {selectedNode.attributes.authors.slice(0, 3).join(", ")}
                              {selectedNode.attributes.authors.length > 3 && ` +${selectedNode.attributes.authors.length - 3} more`}
                            </p>
                          )}
                          {selectedNode.attributes.summary && (
                            <div className="mt-3">
                              <strong className="text-gray-700">Summary:</strong>
                              <div className="mt-1 text-gray-600">
                                <FormattedText text={selectedNode.attributes.summary} />
                              </div>
                            </div>
                          )}
                          <div className="mt-4">
                            <div className="flex items-center justify-between mb-2">
                              <h3 className="font-semibold">Structured Analysis</h3>
                              <button
                                onClick={handleStructuredAnalysis}
                                disabled={isLoadingStructured || !selectedNode.attributes.arxivId}
                                className="px-3 py-1.5 bg-blue-600 text-white rounded font-medium disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
                              >
                                Run Analysis
                              </button>
                            </div>
                            {isLoadingStructured ? (
                              <p className="text-gray-600">Running structured analysis...</p>
                            ) : structuredAnalysis ? (
                              <Accordion type="multiple" className="space-y-2">
                                {structuredAnalysis.sections.map((section, idx) => (
                                  <AccordionItem 
                                    key={`${section.component}-${idx}`} 
                                    value={`section-${idx}`}
                                    className="border border-gray-200 rounded-lg overflow-hidden"
                                  >
                                    <AccordionTrigger className="px-4 py-3 hover:bg-gray-50 font-semibold text-gray-800">
                                      {section.component}
                                    </AccordionTrigger>
                                    <AccordionContent className="px-4 pb-4 space-y-3">
                                      {/* Steps - Indigo */}
                                      <div className="border-l-4 border-indigo-500 pl-3">
                                        <div className="font-medium text-indigo-700 mb-1">Steps</div>
                                        <div className="text-gray-600">
                                          <FormattedText text={section.steps} />
                                        </div>
                                      </div>
                                      {/* Benefits - Emerald */}
                                      <div className="border-l-4 border-emerald-500 pl-3">
                                        <div className="font-medium text-emerald-700 mb-1">Benefits</div>
                                        <div className="text-gray-600">
                                          <FormattedText text={section.benefits} />
                                        </div>
                                      </div>
                                      {/* Rationale - Amber */}
                                      <div className="border-l-4 border-amber-500 pl-3">
                                        <div className="font-medium text-amber-700 mb-1">Rationale</div>
                                        <div className="text-gray-600">
                                          <FormattedText text={section.rationale} />
                                        </div>
                                      </div>
                                      {/* Results - Violet */}
                                      <div className="border-l-4 border-violet-500 pl-3">
                                        <div className="font-medium text-violet-700 mb-1">Results</div>
                                        <div className="text-gray-600">
                                          <FormattedText text={section.results} />
                                        </div>
                                      </div>
                                    </AccordionContent>
                                  </AccordionItem>
                                ))}
                              </Accordion>
                            ) : (
                              <p className="text-gray-500">Run structured analysis to see detailed component breakdowns.</p>
                            )}
                          </div>
                        </div>
                      )}
                      
                    </TabsContent>
                    
                    {/* Details Panel removed - merged into Explorer tab above */}
                    {/* Repos Panel */}
                    <TabsContent value="repos" className="mt-3">
                      <div className="flex items-center justify-between mb-4">
                        <h2 className="font-semibold" style={{ fontSize: `${panelFontSize + 2}px` }}>GitHub Repositories</h2>
                        {selectedNode && selectedNode.attributes?.arxivId && (
                          <button
                            onClick={() => handleFindRepos(selectedNode)}
                            disabled={isLoadingFeature}
                            className="px-3 py-1.5 bg-blue-600 text-white rounded font-medium disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
                          >
                            🔗 Find Repos
                          </button>
                        )}
                      </div>
                      {isLoadingFeature ? (
                        <div className="space-y-3 animate-pulse">
                          <div className="h-16 bg-gray-200 rounded"></div>
                          <div className="h-16 bg-gray-200 rounded"></div>
                          <div className="h-16 bg-gray-200 rounded"></div>
                        </div>
                      ) : repos.length > 0 ? (
                        <div>
                          {repos.map((repo, i) => (
                            <div key={i} className="p-3 border-b border-gray-200">
                              <a href={repo.repo_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 font-medium hover:underline">
                                {repo.repo_name}
                              </a>
                              <div className="text-gray-600 mt-1">
                                {repo.is_official && <span className="text-emerald-600 mr-2">✓ Official</span>}
                                <span>⭐ {repo.stars || 0}</span>
                                <span className="ml-2">via {repo.source}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-500">Click "Find Repos" to search for GitHub repositories related to this paper.</p>
                      )}
                    </TabsContent>

                    {/* References Panel */}
                    <TabsContent value="refs" className="mt-3">
                      <div className="flex items-center justify-between mb-4">
                        <h2 className="font-semibold" style={{ fontSize: `${panelFontSize + 2}px` }}>References</h2>
                        {selectedNode && selectedNode.attributes?.arxivId && (
                          <button
                            onClick={() => handleFetchReferences(selectedNode)}
                            disabled={isLoadingFeature}
                            className="px-3 py-1.5 bg-blue-600 text-white rounded font-medium disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
                          >
                            📚 Explain References
                          </button>
                        )}
                      </div>
                      {isLoadingFeature ? (
                        <p className="text-gray-600">Loading references...</p>
                      ) : references.length > 0 ? (
                        <div>
                          {references.map((ref) => (
                            <div
                              key={ref.id}
                              className="p-3 border-b border-gray-200 relative"
                              onMouseEnter={() => handleRefHover(ref)}
                              onMouseLeave={handleRefLeave}
                            >
                              <div className="font-medium">{ref.cited_title}</div>
                              <div className="text-gray-600 mt-1">
                                {ref.cited_authors?.slice(0, 2).join(", ")}
                                {ref.cited_authors && ref.cited_authors.length > 2 && " et al."}
                                {ref.cited_year && ` (${ref.cited_year})`}
                              </div>
                              {ref.cited_arxiv_id && (
                                <button
                                  onClick={() => handleAddReference(ref)}
                                  className="mt-2 px-2 py-1 bg-gray-100 border-none rounded cursor-pointer hover:bg-gray-200"
                                >
                                  + Add to tree
                                </button>
                              )}
                              
                              {/* Hover tooltip */}
                              {hoveredRefId === ref.id && (
                                <div className="absolute left-full top-0 ml-2 w-[300px] p-3 bg-gray-800 text-white rounded leading-relaxed z-50 shadow-lg">
                                  {refExplanations[ref.id] || "Loading explanation..."}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
              ) : (
                <p className="text-gray-500">Click "Explain References" to load and explain references from this paper.</p>
              )}
                    </TabsContent>

                    {/* Similar Papers Panel */}
                    <TabsContent value="similar" className="mt-3">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h2 className="font-semibold" style={{ fontSize: `${panelFontSize + 2}px` }}>Similar Papers</h2>
                          <p className="text-gray-600 mt-1">
                            Recommended papers from Semantic Scholar (200M+ papers)
                          </p>
                        </div>
                        {selectedNode && selectedNode.attributes?.arxivId && (
                          <button
                            onClick={() => handleFindSimilar(selectedNode)}
                            disabled={isLoadingFeature}
                            className="px-3 py-1.5 bg-blue-600 text-white rounded font-medium disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors whitespace-nowrap"
                          >
                            🔍 Find Similar
                          </button>
                        )}
                      </div>
                      {isLoadingFeature ? (
                        <p className="text-gray-600">Searching the internet for similar papers...</p>
                      ) : similarPapers.length > 0 ? (
                        <div>
                          {similarPapers.map((paper, i) => (
                            <div key={i} className="p-3 border-b border-gray-200">
                              <div className="font-medium">
                                {paper.url ? (
                                  <a href={paper.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                                    {paper.title}
                                  </a>
                                ) : paper.title}
                              </div>
                              <div className="text-gray-600 mt-1">
                                {paper.year && <span>Year: {paper.year}</span>}
                                {paper.citation_count !== undefined && (
                                  <span className="ml-2">Citations: {paper.citation_count}</span>
                                )}
                                {paper.arxiv_id && (
                                  <span className="ml-2">arXiv: {paper.arxiv_id}</span>
                                )}
                              </div>
                              {paper.authors && paper.authors.length > 0 && (
                                <div className="text-gray-500 mt-1">
                                  {paper.authors.slice(0, 3).join(", ")}
                                  {paper.authors.length > 3 && ` +${paper.authors.length - 3} more`}
                                </div>
                              )}
                              {paper.arxiv_id && (
                                <button
                                  onClick={() => handleAddSimilarPaper(paper)}
                                  className="mt-2 px-2 py-1 bg-gray-100 border-none rounded cursor-pointer hover:bg-gray-200"
                                >
                                  + Add to tree
                                </button>
                              )}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-500">Right-click a paper to find similar.</p>
                      )}
                    </TabsContent>

                    {/* Query Panel */}
                    <TabsContent value="query" className="mt-3">
                      <h2 className="font-semibold mb-4" style={{ fontSize: `${panelFontSize + 2}px` }}>Ask a Question</h2>
                      {selectedNode?.attributes?.arxivId ? (
                        <div>
                          <p className="text-gray-600 mb-3">
                            Ask questions about: {selectedNode.attributes.title || selectedNode.name}
                          </p>
                          <textarea
                            value={queryInput}
                            onChange={(e) => setQueryInput(e.target.value)}
                            placeholder="e.g., What is the main contribution of this paper?"
                            disabled={isQuerying}
                            className="w-full h-20 p-2 border border-gray-300 rounded resize-y box-border"
                          />
                          <button
                            onClick={handleQuery}
                            disabled={isQuerying || !queryInput.trim()}
                            className="mt-2 w-full py-2 bg-blue-600 text-white border-none rounded disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-blue-700"
                          >
                            {isQuerying ? "Searching..." : "Ask"}
                          </button>
                          {queryAnswer && (
                            <Card className="mt-4">
                              <CardContent className="pt-4">
                                <h4 className="mb-2 text-gray-600">Answer:</h4>
                                <p className="leading-relaxed whitespace-pre-wrap">
                                  <TextWithMath text={queryAnswer} />
                                </p>
                              </CardContent>
                            </Card>
                          )}
                  
                          {/* Query History */}
                          {queryHistory.length > 0 && (
                            <div className="mt-6">
                              <h3 className="font-semibold mb-3 text-gray-800">
                                Query History ({queryHistory.length})
                                {selectedQueryIds.size > 0 && (
                                  <span className="font-normal text-blue-600 ml-2">
                                    ({selectedQueryIds.size} selected)
                                  </span>
                                )}
                              </h3>
                              {queryHistory.map((q) => (
                                <Card 
                                  key={q.id} 
                                  onClick={() => toggleQuerySelection(q.id)}
                                  className={`mb-3 cursor-pointer transition-all ${
                                    selectedQueryIds.has(q.id) ? "bg-blue-50 border-blue-300" : "bg-gray-50"
                                  }`}
                                >
                                  <CardContent className="pt-4">
                                    <div className="flex items-start gap-2">
                                      <input
                                        type="checkbox"
                                        checked={selectedQueryIds.has(q.id)}
                                        onChange={() => toggleQuerySelection(q.id)}
                                        onClick={(e) => e.stopPropagation()}
                                        className="mt-0.5 cursor-pointer"
                                      />
                                      <div className="flex-1">
                                        <div className="text-gray-600 mb-1">
                                          {new Date(q.created_at).toLocaleString()}
                                        </div>
                                        <div className="font-medium mb-2">
                                          Q: {q.question}
                                        </div>
                                        <div className="text-gray-700 whitespace-pre-wrap">
                                          A: <TextWithMath text={q.answer} />
                                        </div>
                                      </div>
                                    </div>
                                  </CardContent>
                                </Card>
                              ))}
                              
                              {/* Add to Details button */}
                              <button
                                onClick={handleMergeQueries}
                                disabled={selectedQueryIds.size === 0 || isMergingQueries}
                                className="mt-2 w-full py-2 bg-green-600 text-white border-none rounded disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-green-700 font-medium transition-colors"
                              >
                                {isMergingQueries ? "Merging..." : `Add to Details${selectedQueryIds.size > 0 ? ` (${selectedQueryIds.size})` : ""}`}
                              </button>
                            </div>
                          )}
                        </div>
                      ) : (
                        <p className="text-gray-500">Select a paper to ask questions.</p>
                      )}
                    </TabsContent>

                    {/* Topic Query Panel */}
                    <TabsContent value="topic-query" className="mt-3">
                      <h2 className="font-semibold mb-4" style={{ fontSize: `${panelFontSize + 2}px` }}>Topic Queries</h2>
                      
                      {/* Topics List */}
                      {topicList.length === 0 ? (
                        <p className="text-gray-500 mb-4">
                          No topics yet. Use the search box with "Topic" mode to create one.
                        </p>
                      ) : (
                        <div className="space-y-2 mb-4 max-h-96 overflow-y-auto">
                          {topicList.map((topic) => (
                            <div key={topic.id} className="border border-gray-200 rounded">
                              <div
                                className="flex items-center justify-between px-3 py-2 cursor-pointer hover:bg-gray-50"
                                onClick={() => {
                                  if (expandedTopicId === topic.id) {
                                    setExpandedTopicId(null);
                                    setCurrentTopic(null);
                                  } else {
                                    setExpandedTopicId(topic.id);
                                    resumeTopic(topic);
                                  }
                                }}
                              >
                                <div className="flex-1">
                                  <div className="font-medium">{topic.name}</div>
                                  <div className="text-xs text-gray-500">
                                    {topic.paper_count} papers, {topic.query_count} queries
                                  </div>
                                </div>
                                <div className="flex items-center gap-2">
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      deleteTopic(topic.id);
                                    }}
                                    className="text-red-500 hover:text-red-700 text-xs"
                                    title="Delete topic"
                                  >
                                    ✕
                                  </button>
                                  <span className="text-gray-400">
                                    {expandedTopicId === topic.id ? "▼" : "▶"}
                                  </span>
                                </div>
                              </div>
                              
                              {/* Expanded content */}
                              {expandedTopicId === topic.id && currentTopic && (
                                <div className="px-3 py-2 border-t border-gray-200 bg-gray-50">
                                  {/* Papers in pool */}
                                  <div className="mb-3">
                                    <div className="text-sm font-medium mb-1">Papers ({topicPool.length}):</div>
                                    <div className="text-xs text-gray-600 max-h-24 overflow-y-auto">
                                      {topicPool.map((p) => (
                                        <div key={p.paper_id} className="flex items-center justify-between py-0.5">
                                          <span className="truncate flex-1" title={p.title}>{p.title}</span>
                                          <button
                                            onClick={() => removePaperFromPool(p.paper_id)}
                                            className="text-red-400 hover:text-red-600 ml-1"
                                            title="Remove from pool"
                                          >
                                            ✕
                                          </button>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                  
                                  {/* Query history */}
                                  {currentTopic.queries && currentTopic.queries.length > 0 && (
                                    <div className="mb-3">
                                      <div className="text-sm font-medium mb-1">Query History:</div>
                                      <div className="space-y-2 max-h-48 overflow-y-auto">
                                        {currentTopic.queries.map((q) => (
                                          <div key={q.id} className="bg-white p-2 rounded border border-gray-200">
                                            <div className="text-xs text-gray-500">
                                              {new Date(q.created_at).toLocaleString()}
                                            </div>
                                            <div className="font-medium text-sm">Q: {q.question}</div>
                                            <div className="text-sm text-gray-700 mt-1">
                                              <FormattedText text={q.answer} />
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  
                                  {/* Query input */}
                                  <div className="flex gap-2">
                                    <input
                                      type="text"
                                      value={topicQueryInput}
                                      onChange={(e) => setTopicQueryInput(e.target.value)}
                                      onKeyDown={(e) => {
                                        if (e.key === "Enter" && !isLoadingTopicQuery) {
                                          handleTopicQuerySubmit();
                                        }
                                      }}
                                      placeholder="Ask a question about this topic..."
                                      className="flex-1 px-2 py-1 border border-gray-300 rounded text-sm"
                                      disabled={isLoadingTopicQuery || topicPool.length === 0}
                                    />
                                    <button
                                      onClick={handleTopicQuerySubmit}
                                      disabled={isLoadingTopicQuery || !topicQueryInput.trim() || topicPool.length === 0}
                                      className="px-3 py-1 bg-blue-600 text-white rounded text-sm disabled:bg-gray-400"
                                    >
                                      {isLoadingTopicQuery ? "..." : "Ask"}
                                    </button>
                                  </div>
                                  {topicPool.length === 0 && (
                                    <p className="text-xs text-amber-600 mt-1">Add papers to the pool first.</p>
                                  )}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                      
                      <p className="text-xs text-gray-500">
                        Tip: Select "Topic" in the search dropdown and press Enter to create a new topic.
                      </p>
                    </TabsContent>
                  </Tabs>
                    )}
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </Card>
        
        {/* Debug Panel section removed - now integrated above tabs */}
      </div>
      
      {/* Topic Dialog for paper selection */}
      {showTopicDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden flex flex-col">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold">
                {topicDialogMode === "select" ? "Topic Exists" : 
                 topicDialogMode === "create" ? "Create New Topic" :
                 topicDialogMode === "building" ? "Build Paper Pool" : "Checking..."}
              </h3>
            </div>
            
            <div className="px-6 py-4 flex-1 overflow-y-auto">
              {/* Select existing or create new */}
              {topicDialogMode === "select" && (
                <div>
                  <p className="mb-4 text-gray-600">
                    Topics already exist for "{currentTopicQuery}". Choose an option:
                  </p>
                  
                  <div className="space-y-2 mb-4">
                    {existingTopics.map((topic) => (
                      <div
                        key={topic.id}
                        className="border border-gray-200 rounded p-3 hover:bg-gray-50 cursor-pointer"
                        onClick={() => resumeTopic(topic)}
                      >
                        <div className="font-medium">{topic.name}</div>
                        <div className="text-sm text-gray-500">
                          {topic.paper_count} papers, {topic.query_count} queries
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="border-t border-gray-200 pt-4">
                    <p className="text-sm text-gray-600 mb-2">Or create a new topic with a prefix:</p>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={topicNamePrefix}
                        onChange={(e) => setTopicNamePrefix(e.target.value)}
                        placeholder="Enter prefix (e.g., 'v2' or 'detailed')"
                        className="flex-1 px-3 py-2 border border-gray-300 rounded"
                      />
                      <button
                        onClick={() => {
                          if (topicNamePrefix.trim()) {
                            startNewTopic(currentTopicQuery, topicNamePrefix.trim());
                          }
                        }}
                        disabled={!topicNamePrefix.trim()}
                        className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-gray-400"
                      >
                        Create
                      </button>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Building paper pool */}
              {topicDialogMode === "building" && (
                <div>
                  <p className="mb-2 text-gray-600">
                    Topic: <strong>{currentTopic?.name || currentTopicQuery}</strong>
                  </p>
                  
                  {/* Current pool */}
                  {topicPool.length > 0 && (
                    <div className="mb-4 p-3 bg-blue-50 rounded">
                      <div className="text-sm font-medium mb-1">Papers in pool ({topicPool.length}):</div>
                      <div className="max-h-24 overflow-y-auto">
                        {topicPool.map((p) => (
                          <div key={p.paper_id} className="flex items-center justify-between text-sm py-0.5">
                            <span className="truncate flex-1">{p.title}</span>
                            <span className="text-gray-500 mx-2">{(p.similarity * 100).toFixed(0)}%</span>
                            <button
                              onClick={() => removePaperFromPool(p.paper_id)}
                              className="text-red-500 hover:text-red-700"
                            >
                              ✕
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Paper selection */}
                  <div className="mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Select papers to add:</span>
                      <div className="flex gap-2">
                        <button
                          onClick={() => setTopicSearchResults(r => r.map(p => ({ ...p, selected: true })))}
                          className="text-xs text-blue-600 hover:underline"
                        >
                          Select All
                        </button>
                        <button
                          onClick={() => setTopicSearchResults(r => r.map(p => ({ ...p, selected: false })))}
                          className="text-xs text-blue-600 hover:underline"
                        >
                          Select None
                        </button>
                      </div>
                    </div>
                    
                    {isLoadingTopicSearch ? (
                      <p className="text-gray-500 py-4 text-center">Loading papers...</p>
                    ) : topicSearchResults.length === 0 ? (
                      <p className="text-gray-500 py-4 text-center">No more papers found above threshold.</p>
                    ) : (
                      <div className="max-h-48 overflow-y-auto border border-gray-200 rounded">
                        {topicSearchResults.map((paper) => (
                          <div
                            key={paper.paper_id}
                            className={`flex items-center gap-2 px-3 py-2 border-b border-gray-100 last:border-b-0 cursor-pointer hover:bg-gray-50 ${paper.selected ? 'bg-blue-50' : ''}`}
                            onClick={() => {
                              setTopicSearchResults(r => r.map(p => 
                                p.paper_id === paper.paper_id ? { ...p, selected: !p.selected } : p
                              ));
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={paper.selected || false}
                              onChange={() => {}}
                              className="cursor-pointer"
                            />
                            <div className="flex-1 min-w-0">
                              <div className="font-medium truncate">{paper.title}</div>
                              <div className="text-xs text-gray-500">
                                {paper.year && `${paper.year} · `}
                                Similarity: {(paper.similarity * 100).toFixed(0)}%
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  {/* Actions */}
                  <div className="flex gap-2">
                    <button
                      onClick={addSelectedPapersToPool}
                      disabled={!topicSearchResults.some(p => p.selected)}
                      className="flex-1 py-2 bg-blue-600 text-white rounded disabled:bg-gray-400"
                    >
                      Add Selected to Pool
                    </button>
                    <button
                      onClick={loadMoreTopicPapers}
                      disabled={isLoadingTopicSearch}
                      className="px-4 py-2 border border-gray-300 rounded hover:bg-gray-50"
                    >
                      Load More
                    </button>
                  </div>
                </div>
              )}
            </div>
            
            <div className="px-6 py-4 border-t border-gray-200 flex justify-between">
              <button
                onClick={() => {
                  setShowTopicDialog(false);
                  setTopicSearchResults([]);
                  setTopicNamePrefix("");
                }}
                className="px-4 py-2 border border-gray-300 rounded hover:bg-gray-50"
              >
                Cancel
              </button>
              {topicDialogMode === "building" && topicPool.length > 0 && (
                <button
                  onClick={finishPaperSelection}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                >
                  Done ({topicPool.length} papers)
                </button>
              )}
            </div>
          </div>
        </div>
      )}
      
      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden flex flex-col">
            <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
              <h3 className="text-lg font-semibold">Settings</h3>
              <button
                onClick={() => setShowSettings(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            
            <div className="px-6 py-4 flex-1 overflow-y-auto">
              {settingsLoading ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
                </div>
              ) : settingsData ? (
                <div className="space-y-6">
                  {/* Warnings */}
                  {settingsWarnings.length > 0 && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded p-3">
                      <div className="font-medium text-yellow-800 mb-1">Warnings</div>
                      {settingsWarnings.map((w, i) => (
                        <div key={i} className="text-sm text-yellow-700">{w}</div>
                      ))}
                    </div>
                  )}
                  
                  {/* Settings by category */}
                  {Object.entries(settingsData.categories || {}).map(([catKey, catInfo]: [string, any]) => (
                    <div key={catKey} className="border border-gray-200 rounded-lg overflow-hidden">
                      <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
                        <h4 className="font-semibold">{catInfo.label}</h4>
                        <p className="text-xs text-gray-500">{catInfo.description}</p>
                      </div>
                      <div className="p-4 space-y-3">
                        {Object.entries(settingsData.settings || {})
                          .filter(([_, s]: [string, any]) => s.category === catKey)
                          .map(([key, setting]: [string, any]) => (
                            <div key={key} className="flex items-center justify-between gap-4">
                              <div className="flex-1">
                                <label className="text-sm font-medium text-gray-700">
                                  {setting.label}
                                  {setting.is_overridden && (
                                    <span className="ml-2 text-xs text-blue-600">(customized)</span>
                                  )}
                                </label>
                              </div>
                              <div className="w-64">
                                {setting.readonly ? (
                                  <div className="flex items-center gap-2">
                                    <input
                                      type="text"
                                      value={setting.value}
                                      disabled
                                      className="w-full px-3 py-1.5 border border-gray-200 rounded bg-gray-100 text-gray-500 text-sm"
                                    />
                                    <span className="text-xs text-gray-400" title="Read-only. Change in server config.">🔒</span>
                                  </div>
                                ) : setting.type === "boolean" ? (
                                  <select
                                    value={setting.value ? "true" : "false"}
                                    onChange={(e) => handleSaveSettings({ [key]: e.target.value === "true" })}
                                    disabled={settingsSaving}
                                    className="w-full px-3 py-1.5 border border-gray-300 rounded text-sm"
                                  >
                                    <option value="true">Enabled</option>
                                    <option value="false">Disabled</option>
                                  </select>
                                ) : (
                                  <input
                                    type={setting.type === "integer" || setting.type === "float" ? "number" : "text"}
                                    step={setting.type === "float" ? "0.01" : "1"}
                                    defaultValue={setting.value}
                                    onBlur={(e) => {
                                      const newVal = setting.type === "integer" ? parseInt(e.target.value) :
                                                     setting.type === "float" ? parseFloat(e.target.value) :
                                                     e.target.value;
                                      if (newVal !== setting.value) {
                                        handleSaveSettings({ [key]: newVal });
                                      }
                                    }}
                                    disabled={settingsSaving}
                                    className="w-full px-3 py-1.5 border border-gray-300 rounded text-sm"
                                  />
                                )}
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500">Failed to load settings.</p>
              )}
            </div>
            
            <div className="px-6 py-4 border-t border-gray-200 flex justify-between">
              <button
                onClick={handleResetSettings}
                disabled={settingsSaving}
                className="px-4 py-2 text-red-600 border border-red-300 rounded hover:bg-red-50 disabled:opacity-50"
              >
                Reset to Defaults
              </button>
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
    </TooltipProvider>
  );
}
