"""Async post-order contrastive naming for tree nodes.

This module provides LLM-powered naming for the hierarchical paper classification tree.
It uses depth-layered parallel execution to ensure children are named before parents,
enabling proper contrastive context while preserving parallelism.

Key features:
- One-time O(n) indexing of tree structure
- Depth-layered parallelism (children complete before parent)
- Semaphore-controlled concurrency for LLM calls
- Contrastive naming using sibling names + sibling children names

Tree structure expected:
{
    "name": "AI Papers",
    "node_id": "root",
    "node_type": "category",
    "children": [
        {
            "name": "node_xxx",  # Placeholder, will be named by LLM
            "node_id": "node_xxx",
            "node_type": "category",
            "children": [...]
        },
        {
            "name": "paper_123",  # Paper names not touched by naming process
            "node_id": "node_yyy",
            "node_type": "paper",
            "paper_id": 123
        }
    ]
}
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import db

logger = logging.getLogger(__name__)

# =============================================================================
# Prompt Loading
# =============================================================================

def _get_prompt(prompt_name: str, **kwargs: Any) -> str:
    """Load prompt template from prompts.json."""
    import json
    from pathlib import Path
    
    prompt_paths = [
        Path("prompts/prompts.json"),
        Path("../prompts/prompts.json"),
        Path("../../prompts/prompts.json"),
    ]
    
    for path in prompt_paths:
        if path.exists():
            with open(path, "r") as f:
                prompts = json.load(f)
                if prompt_name in prompts:
                    template = prompts[prompt_name]["template"]
                    for key, value in kwargs.items():
                        template = template.replace(f"{{{key}}}", str(value))
                    return template
    
    raise ValueError(f"Prompt '{prompt_name}' not found")


# =============================================================================
# Tree Indexing (One-Time O(n) Pass)
# =============================================================================

class TreeIndex:
    """One-time index of tree structure for O(1) lookups."""
    
    def __init__(self, tree: dict[str, Any], paper_cache: dict[int, dict[str, Any]] | None = None):
        """Build index from tree.
        
        Args:
            tree: Root node of tree structure
            paper_cache: Pre-loaded paper data keyed by paper_id.
                         If provided, avoids N+1 DB queries during indexing.
                         Use db.get_papers_lightweight() to build this.
        """
        # Direct references to node dicts
        self.node_ref: dict[str, dict[str, Any]] = {}
        
        # Parent relationships
        self.parent_of: dict[str, str | None] = {}
        
        # Children relationships
        self.children_of: dict[str, list[str]] = {}
        
        # Depth of each node
        self.depth_of: dict[str, int] = {}
        
        # Nodes grouped by depth (for bottom-up ordering)
        self.nodes_at_depth: dict[int, list[str]] = {}
        
        # Max depth
        self.max_depth = 0
        
        # Paper summaries cache (paper_id -> summary)
        self.paper_summaries: dict[int, str] = {}
        
        # Pre-loaded paper data (avoids N+1 queries)
        self._paper_cache = paper_cache or {}
        
        # Build index
        self._index_node(tree, parent_id=None, depth=0)
    
    def _index_node(self, node: dict[str, Any], parent_id: str | None, depth: int) -> None:
        """Recursively index a node and its children."""
        node_id = node.get("node_id")
        if not node_id:
            # Root might not have node_id, use "root"
            node_id = "root"
            node["node_id"] = node_id
        
        # Ensure root has node_type
        if not node.get("node_type") and node.get("children"):
            node["node_type"] = "category"
        
        # Store references
        self.node_ref[node_id] = node
        self.parent_of[node_id] = parent_id
        self.depth_of[node_id] = depth
        self.max_depth = max(self.max_depth, depth)
        
        # Group by depth
        if depth not in self.nodes_at_depth:
            self.nodes_at_depth[depth] = []
        self.nodes_at_depth[depth].append(node_id)
        
        # Index children
        children = node.get("children", [])
        child_ids = []
        for child in children:
            child_id = child.get("node_id")
            if child_id:
                child_ids.append(child_id)
                self._index_node(child, parent_id=node_id, depth=depth + 1)
        
        self.children_of[node_id] = child_ids
        
        # Cache paper summary if this is a paper node
        if node.get("node_type") == "paper" and node.get("paper_id"):
            paper = self._paper_cache.get(node["paper_id"])
            if paper and paper.get("summary"):
                self.paper_summaries[node["paper_id"]] = paper["summary"]
    
    def get_siblings(self, node_id: str) -> list[str]:
        """Get sibling node IDs (same parent, excluding self)."""
        parent_id = self.parent_of.get(node_id)
        if not parent_id:
            return []
        return [s for s in self.children_of.get(parent_id, []) if s != node_id]
    
    def get_category_nodes_bottom_up(self) -> list[str]:
        """Get all category node IDs ordered from deepest to shallowest."""
        result = []
        for depth in range(self.max_depth, -1, -1):
            for node_id in self.nodes_at_depth.get(depth, []):
                node = self.node_ref.get(node_id)
                if node and node.get("node_type") == "category":
                    result.append(node_id)
        return result
    
    def is_leaf_category(self, node_id: str) -> bool:
        """Check if a category node's children are all papers."""
        node = self.node_ref.get(node_id)
        if not node or node.get("node_type") != "category":
            return False
        children = node.get("children", [])
        if not children:
            return True
        return all(c.get("node_type") == "paper" for c in children)
    
    def get_paper_summary(self, paper_id: int) -> str:
        """Get cached paper summary."""
        return self.paper_summaries.get(paper_id, "")
    
    def get_paper_abbreviation(self, paper_id: int) -> str:
        """Get paper abbreviation from title, or fallback to paper_<id>."""
        paper = self._paper_cache.get(paper_id)
        if paper and paper.get("title"):
            title = paper["title"]
            words = title.split()
            # Use first 2-3 words
            abbrev = " ".join(words[:3])
            if len(abbrev) > 20:
                abbrev = " ".join(words[:2])
            if len(abbrev) > 20:
                abbrev = words[0][:18] + ".."
            elif len(words) > 3:
                abbrev += ".."
            return abbrev
        return f"paper_{paper_id}"


# =============================================================================
# Async Naming with Future-Based Coordination
# =============================================================================

class AsyncTreeNamer:
    """Async naming engine with depth-layered parallel execution."""
    
    def __init__(
        self,
        tree: dict[str, Any],
        llm_client: Any,
        model: str,
        max_concurrent: int = 5,
        debug: bool = False,
        paper_cache: dict[int, dict[str, Any]] | None = None,
    ):
        """Initialize namer.
        
        Args:
            tree: Root node of tree structure
            llm_client: AsyncOpenAI client
            model: Model name for LLM calls
            max_concurrent: Max concurrent LLM calls
            debug: If True, save LLM calls to schemas/llm_naming.json
            paper_cache: Pre-loaded paper data (from db.get_papers_lightweight)
        """
        self.tree = tree
        self.index = TreeIndex(tree, paper_cache=paper_cache)
        self.client = llm_client
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.debug = debug
        
        self.nodes_named = 0
        
        # Debug: store LLM calls
        self.llm_calls: dict[str, dict[str, Any]] = {}
        
        # Ensure paper nodes have abbreviations
        self._init_paper_names()

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Normalize LLM names by removing markdown and boilerplate."""
        cleaned = name.strip().strip('"\'')
        if not cleaned:
            return cleaned
        cleaned = cleaned.splitlines()[0].strip()
        cleaned = re.sub(r"[*_`]+", "", cleaned)
        cleaned = re.sub(r"^\d+[\).\s]+", "", cleaned)
        cleaned = re.sub(r"^[#>\-\s]+", "", cleaned)
        if ":" in cleaned:
            prefix, rest = cleaned.split(":", 1)
            if rest.strip() and len(prefix.strip()) <= 20:
                cleaned = rest.strip()
            else:
                cleaned = cleaned.rstrip(":").strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned
    
    def _init_paper_names(self) -> None:
        """Set paper node names using abbreviation or fallback."""
        for node in self.index.node_ref.values():
            if node.get("node_type") != "paper":
                continue
            paper_id = node.get("paper_id")
            if not paper_id:
                continue
            abbrev = self.index.get_paper_abbreviation(paper_id)
            if not node.get("name") or node.get("name", "").startswith("paper_"):
                node["name"] = abbrev
    
    async def name_all_categories(self) -> dict[str, Any]:
        """Name all category nodes using depth-layered parallel execution.
        
        Returns:
            Results dictionary with stats
        """
        import json
        import os
        
        # Get all category nodes grouped by depth
        if not self.index.nodes_at_depth:
            return {
                "nodes_named": 0,
                "message": "No category nodes to name",
            }
        
        category_nodes = self.index.get_category_nodes_bottom_up()
        print(f"Naming {len(category_nodes)} category nodes...")
        
        # Process depth by depth (deepest first). Nodes at the same depth
        # can be safely named in parallel because their children are deeper.
        for depth in range(self.index.max_depth, -1, -1):
            level_nodes = [
                node_id
                for node_id in self.index.nodes_at_depth.get(depth, [])
                if self.index.node_ref.get(node_id, {}).get("node_type") == "category"
            ]
            if not level_nodes:
                continue
            
            tasks = [self._name_node(node_id) for node_id in level_nodes]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for node_id, result in zip(level_nodes, results):
                if isinstance(result, Exception):
                    print(f"  ✗ Error naming {node_id}: {result}")
        
        # Debug: save LLM calls to file (use storage dir for Singularity compatibility)
        if self.debug and self.llm_calls:
            os.makedirs("storage/schemas", exist_ok=True)
            with open("storage/schemas/llm_naming.json", "w") as f:
                json.dump(self.llm_calls, f, indent=2)
            print(f"Debug: Saved {len(self.llm_calls)} LLM calls to storage/schemas/llm_naming.json")
        
        return {
            "nodes_named": self.nodes_named,
            "total_categories": len(category_nodes),
            "message": f"Named {self.nodes_named} of {len(category_nodes)} category nodes",
        }
    
    async def _name_node(self, node_id: str) -> str:
        """Name a single category node.
        
        Generates name using contrastive context (sibling names + sibling
        children names). Children are already named because this runs in
        depth-layered order.
        
        Args:
            node_id: Node ID to name
            
        Returns:
            The generated name
        """
        node = self.index.node_ref.get(node_id)
        if not node:
            return node_id
        
        # Skip if not a category
        if node.get("node_type") != "category":
            return node.get("name", node_id)
        
        # Skip root node (keep "AI Papers")
        if node_id == "root":
            return node.get("name", "AI Papers")
        
        # 1. Collect child names (already named at deeper levels)
        child_ids = self.index.children_of.get(node_id, [])
        child_names = [
            self.index.node_ref.get(cid, {}).get("name", cid)
            for cid in child_ids
        ]
        
        # 2. Get sibling context (sibling names + sibling children names)
        sibling_ids = self.index.get_siblings(node_id)
        sibling_context: list[str] = []
        
        for sib_id in sibling_ids:
            sib_name = self.index.node_ref.get(sib_id, {}).get("name", sib_id)
            sibling_context.append(f"Sibling: {sib_name}")
            
            sib_child_ids = self.index.children_of.get(sib_id, [])
            for sib_child_id in sib_child_ids:
                sib_child_name = self.index.node_ref.get(sib_child_id, {}).get("name", sib_child_id)
                sibling_context.append(f"  - {sib_child_name}")
        
        # 3. Build children content
        is_leaf = self.index.is_leaf_category(node_id)
        if is_leaf:
            # Use paper summaries for leaf categories
            children_content = []
            for child in node.get("children", []):
                if child.get("paper_id"):
                    summary = self.index.get_paper_summary(child["paper_id"])
                    if summary:
                        children_content.append(summary[:500])  # Truncate
        else:
            # Use child category names
            children_content = list(child_names)
        
        # 4. Call LLM with semaphore
        async with self.semaphore:
            new_name = await self._call_llm_for_name(
                children_content=children_content,
                sibling_context=sibling_context,
                node_id=node_id,
            )
        
        # 5. Update node
        node["name"] = new_name
        db.update_tree_node_name(node_id, new_name)
        
        self.nodes_named += 1
        print(f"  ✓ Named {node_id} as '{new_name}'")
        
        return new_name
    
    async def _call_llm_for_name(
        self,
        children_content: list[str],
        sibling_context: list[str],
        node_id: str,
    ) -> str:
        """Call LLM to generate a contrastive name.
        
        Args:
            children_content: List of children summaries or names
            sibling_context: List of sibling info strings
            node_id: Node ID (for fallback)
            
        Returns:
            Generated name
        """
        children_text = "\n\n".join([
            f"Child {i+1}:\n{content}"
            for i, content in enumerate(children_content)
        ]) if children_content else "No children content available"
        
        siblings_text = "\n".join(sibling_context) if sibling_context else "None"
        
        max_retries = 3
        prompt = None
        response_text = None
        
        for attempt in range(max_retries):
            try:
                prompt = _get_prompt(
                    "node_naming",
                    children_summaries=children_text,
                    sibling_summaries=siblings_text,
                )
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.7,  # Higher temperature for variety on re-naming
                )
                if not response.choices:
                    raise RuntimeError(f"LLM returned empty response for node {node_id}")
                response_text = response.choices[0].message.content
                raw_name = response_text.strip().strip('"\'')
                new_name = self._sanitize_name(raw_name)
                
                # Debug: log LLM call
                if self.debug:
                    self.llm_calls[node_id] = {
                        "prompt": prompt,
                        "response": response_text,
                        "raw_name": raw_name,
                        "name": new_name,
                        "model": self.model,
                        "attempt": attempt + 1,
                        "status": "success",
                    }
                
                # Validate name
                invalid_starts = (
                    "what it does",
                    "overview",
                    "general",
                    "misc",
                    "miscellaneous",
                    "summary",
                )
                lowered = new_name.lower()
                invalid = any(lowered.startswith(prefix) for prefix in invalid_starts)
                if new_name and len(new_name) > 3 and not new_name.startswith("Category_") and not invalid:
                    return new_name
                else:
                    raise ValueError(f"Invalid name: '{new_name}'")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    # Fallback: use first child content words
                    fallback = node_id
                    if children_content:
                        words = children_content[0].split()[:3]
                        fallback = " ".join(words).title()[:30] or node_id
                    fallback = self._sanitize_name(fallback)
                    
                    # Debug: log failed LLM call
                    if self.debug:
                        self.llm_calls[node_id] = {
                            "prompt": prompt,
                            "response": response_text,
                            "name": fallback,
                            "model": self.model,
                            "attempt": attempt + 1,
                            "status": "fallback",
                            "error": str(e),
                        }
                    
                    return fallback
        
        return node_id


# =============================================================================
# Main Entry Point
# =============================================================================

async def name_tree_nodes(debug: bool = False) -> dict[str, Any]:
    """Name all tree nodes using async post-order contrastive naming.
    
    Process:
    1. Load tree and build one-time index
    2. Initialize paper abbreviations
    3. Launch category naming tasks depth by depth
    4. Each task calls LLM with contrastive context
    5. Update names in tree and database
    
    Args:
        debug: If True, save LLM calls to schemas/llm_naming.json
    
    Returns:
        Dictionary with results
    """
    # Import LLM helpers from shared modules
    from config import _get_endpoint_config
    from llm_clients import resolve_model as _resolve_model
    from llm_clients import get_async_openai_client as _get_async_openai_client
    
    # Load tree
    tree = db.get_tree()
    
    # Get LLM config
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    print(f"Starting async tree naming with model: {model}")
    if debug:
        print("Debug mode: LLM calls will be saved to schemas/llm_naming.json")
    
    # Bulk-load paper data in a single query (avoids 4000+ individual DB calls)
    paper_cache = db.get_papers_lightweight()
    print(f"Loaded {len(paper_cache)} papers for tree naming")
    
    # Create namer and run
    namer = AsyncTreeNamer(
        tree=tree,
        llm_client=client,
        model=model,
        max_concurrent=5,
        debug=debug,
        paper_cache=paper_cache,
    )
    
    result = await namer.name_all_categories()
    
    # Enforce unique node names by appending a deterministic node_id suffix
    def build_node_ref(node: dict, mapping: dict[str, dict]) -> None:
        node_id = node.get("node_id")
        if node_id:
            mapping[node_id] = node
        for child in node.get("children", []):
            build_node_ref(child, mapping)
    
    node_ref: dict[str, dict] = {}
    build_node_ref(tree, node_ref)

    raw_name_map = {node_id: (node.get("name") or "") for node_id, node in node_ref.items()}
    
    name_to_ids: dict[str, list[str]] = {}
    for node_id, node in node_ref.items():
        name = (node.get("name") or "").strip()
        if name:
            name_to_ids.setdefault(name, []).append(node_id)
    
    for name, node_ids in name_to_ids.items():
        if len(node_ids) <= 1:
            continue
        for node_id in node_ids:
            suffix = node_id[-6:] if node_id else "unknown"
            new_name = f"{name} ({suffix})"
            node_ref[node_id]["name"] = new_name
            db.update_tree_node_name(node_id, new_name)
    
    # Save updated tree to database
    db.save_tree(tree)
    
    # Export named tree to JSON for verification (includes raw names)
    import json
    import os

    def simplify_node(node: dict) -> dict:
        r = {}
        node_id = node.get("node_id")
        if node_id:
            r["node_id"] = node_id
            r["raw_name"] = raw_name_map.get(node_id, "")
        r["name"] = node.get("name", "")
        r["node_type"] = node.get("node_type", "category")
        if node.get("paper_id"):
            r["paper_id"] = node["paper_id"]
        if node.get("children"):
            r["children"] = [simplify_node(c) for c in node.get("children", [])]
        return r

    # Use storage directory which is writable in Singularity containers
    os.makedirs("storage/schemas", exist_ok=True)
    with open("storage/schemas/named_tree.json", "w") as f:
        json.dump(simplify_node(tree), f, indent=2)
    if debug:
        print("Debug: Saved named tree to storage/schemas/named_tree.json")
    
    return result


# =============================================================================
# Single Node Rename (For UI Re-abbreviation)
# =============================================================================

async def rename_single_category(
    node_id: str,
    custom_name: str | None = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Rename a single category node using LLM or custom name.
    
    This function is used by the frontend's "Re-abbreviate" button for categories.
    It uses the same contrastive naming logic as the full tree naming process.
    
    Args:
        node_id: The category node ID to rename
        custom_name: If provided, use this name directly (skip LLM)
        temperature: LLM temperature (default 0.7 for variety)
    
    Returns:
        dict with 'name', 'node_id', 'used_custom'
    """
    from llm_clients import get_async_openai_client, get_llm_config
    
    # Get tree and build index
    tree = db.get_tree()
    
    # Find the node
    def find_node(node: dict, target_id: str) -> dict | None:
        if node.get("node_id") == target_id:
            return node
        for child in node.get("children", []):
            found = find_node(child, target_id)
            if found:
                return found
        return None
    
    target_node = find_node(tree, node_id)
    if not target_node:
        raise ValueError(f"Node {node_id} not found")
    
    if target_node.get("node_type") == "paper":
        raise ValueError("Use /papers/reabbreviate for paper nodes")
    
    # If custom name provided, use it directly
    if custom_name:
        new_name = custom_name.strip()
        db.update_tree_node_name(node_id, new_name)
        return {"name": new_name, "node_id": node_id, "used_custom": True}
    
    # Build context using tree index
    index = TreeIndex(tree)
    
    # Get child names
    child_ids = index.children_of.get(node_id, [])
    child_names = [
        index.node_ref.get(cid, {}).get("name", cid)
        for cid in child_ids
    ]
    
    # Get sibling context
    sibling_ids = index.get_siblings(node_id)
    sibling_context: list[str] = []
    
    for sib_id in sibling_ids:
        sib_name = index.node_ref.get(sib_id, {}).get("name", sib_id)
        sibling_context.append(f"Sibling: {sib_name}")
        
        sib_child_ids = index.children_of.get(sib_id, [])
        for sib_child_id in sib_child_ids:
            sib_child_name = index.node_ref.get(sib_child_id, {}).get("name", sib_child_id)
            sibling_context.append(f"  - {sib_child_name}")
    
    # Build children content
    is_leaf = index.is_leaf_category(node_id)
    if is_leaf:
        # Use paper summaries for leaf categories
        children_content = []
        for child in target_node.get("children", []):
            if child.get("paper_id"):
                summary = index.get_paper_summary(child["paper_id"])
                if summary:
                    children_content.append(summary[:500])
    else:
        # Use child category names
        children_content = list(child_names)
    
    # Call LLM
    llm_config = get_llm_config()
    client = get_async_openai_client(llm_config["base_url"], llm_config["api_key"])
    model = llm_config["model"]
    
    children_text = "\n\n".join([
        f"Child {i+1}:\n{content}"
        for i, content in enumerate(children_content)
    ]) if children_content else "No children content available"
    
    siblings_text = "\n".join(sibling_context) if sibling_context else "None"
    
    prompt = _get_prompt(
        "node_naming",
        children_summaries=children_text,
        sibling_summaries=siblings_text,
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=temperature,
    )
    if not response.choices:
        raise HTTPException(status_code=502, detail="LLM returned empty response (choices=[])")
    raw_name = response.choices[0].message.content.strip().strip('"\'')
    new_name = AsyncTreeNamer._sanitize_name(raw_name)
    
    # Update database
    db.update_tree_node_name(node_id, new_name)
    
    return {"name": new_name, "node_id": node_id, "used_custom": False}


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    """
    Standalone script to export the current tree as a JSON file for verification.
    
    Usage:
        python naming.py
    
    Output:
        schemas/named_tree.json - The full tree structure with node_id, node_name, etc.
    """
    import json
    import os
    
    # Get the current tree from database
    tree = db.get_tree()
    
    def simplify_node(node: dict) -> dict:
        """Extract only essential fields for verification."""
        result = {}
        
        if node.get("node_id"):
            result["node_id"] = node["node_id"]
        result["name"] = node.get("name", "")
        result["node_type"] = node.get("node_type", "category")
        
        if node.get("paper_id"):
            result["paper_id"] = node["paper_id"]
        if node.get("attributes", {}).get("title"):
            result["title"] = node["attributes"]["title"]
        
        if node.get("children"):
            result["children"] = [simplify_node(child) for child in node["children"]]
        
        return result
    
    simplified_tree = simplify_node(tree)
    
    # Count nodes
    all_node_ids = []
    counts = {"paper": 0, "category": 0}
    
    def count_nodes(node: dict):
        if node.get("node_id"):
            all_node_ids.append(node["node_id"])
        if node.get("node_type") == "paper":
            counts["paper"] += 1
        else:
            counts["category"] += 1
        for child in node.get("children", []):
            count_nodes(child)
    
    count_nodes(simplified_tree)
    
    # Check for duplicates
    from collections import Counter
    id_counts = Counter(all_node_ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1}
    
    # Save (use storage dir for Singularity compatibility)
    os.makedirs("storage/schemas", exist_ok=True)
    output_path = "storage/schemas/named_tree.json"
    with open(output_path, "w") as f:
        json.dump(simplified_tree, f, indent=2)
    
    print(f"Tree exported to {output_path}")
    print(f"  Total nodes: {len(all_node_ids)}")
    print(f"  Categories: {counts['category']}")
    print(f"  Papers: {counts['paper']}")
    print(f"  Unique node_ids: {len(set(all_node_ids))}")
    
    if duplicates:
        print(f"\n⚠️  DUPLICATE NODE IDs FOUND: {len(duplicates)}")
        for node_id, count in duplicates.items():
            print(f"    {node_id}: appears {count} times")
    else:
        print("\n✓ All node_ids are unique")
