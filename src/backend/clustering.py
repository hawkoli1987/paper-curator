"""Hierarchical clustering for paper classification using document embeddings.

This module builds a hierarchical tree structure from paper embeddings using
divisive k-means clustering. The tree is built directly in frontend-compatible
format and can be saved to the database as JSONB.

Key features:
- L2-normalized embeddings for cosine-similarity-based clustering
- Adaptive k selection using silhouette scoring
- BisectingKMeans fallback for robust cluster splitting
- Direct frontend format output (no wrapping step)
"""

from __future__ import annotations

import hashlib
import json
import logging
import numpy as np
from typing import Any
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

import db

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

def _get_clustering_config() -> dict[str, Any]:
    """Get clustering configuration from config (with DB overrides).
    
    Returns:
        Dictionary with clustering parameters:
        - branching_factor: Maximum children before branching
        - clustering_method: Clustering algorithm (currently only "divisive")
    """
    from config import _get_classification_config
    
    # Get config from centralized config module (supports DB overrides)
    classification_config = _get_classification_config()
    
    return {
        "branching_factor": classification_config["branching_factor"],
        "clustering_method": "divisive",  # Currently only divisive is supported
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_embedding_to_numpy(embedding) -> np.ndarray:
    """Convert pgvector embedding to numpy array.
    
    Args:
        embedding: Embedding from database (pgvector type)
        
    Returns:
        Numpy array of shape (embedding_dim,)
    """
    if hasattr(embedding, 'tolist'):
        emb_list = embedding.tolist()
    elif isinstance(embedding, (list, tuple)):
        emb_list = list(embedding)
    else:
        emb_list = list(embedding)
    
    return np.array(emb_list, dtype=np.float32)


def _generate_node_id(paper_ids: list[int]) -> str:
    """Generate a deterministic node ID from sorted paper IDs using stable hash.
    
    Args:
        paper_ids: List of paper IDs in this node (will be sorted for stability)
        
    Returns:
        Deterministic node ID string with 'node_' prefix
    """
    sorted_ids = sorted(paper_ids)
    id_string = ",".join(str(pid) for pid in sorted_ids)
    hash_obj = hashlib.sha256(id_string.encode())
    unique_suffix = hash_obj.hexdigest()[:12]
    return f"node_{unique_suffix}"


# =============================================================================
# Clustering Algorithm Helpers
# =============================================================================

def _check_embeddings_valid(embeddings: np.ndarray, k: int) -> bool:
    """Check if embeddings are valid for k-means clustering.
    
    Args:
        embeddings: Embeddings array (should be L2-normalized)
        k: Number of clusters to create
        
    Returns:
        True if valid, False if should fallback to leaf
    """
    # Check if enough papers for k clusters
    if embeddings.shape[0] < k:
        return False
    
    # Check if all embeddings are identical (would cause k-means to fail)
    if embeddings.shape[0] > 1:
        embedding_std = np.std(embeddings, axis=0)
        if np.all(embedding_std == 0):
            return False
    
    return True


def _compute_silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score for clustering quality using sklearn.
    
    Since embeddings are L2-normalized, Euclidean distance is equivalent to
    cosine distance (up to a constant factor).
    
    Args:
        embeddings: L2-normalized embeddings array
        labels: Cluster labels from k-means
        
    Returns:
        Silhouette score (higher is better, range: -1 to 1)
    """
    n_samples = len(labels)
    if n_samples < 2:
        return 0.0
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    try:
        # Use Euclidean on normalized embeddings (equivalent to cosine)
        return silhouette_score(embeddings, labels, metric='euclidean')
    except Exception as e:
        logger.warning(f"Silhouette score computation failed: {e}")
        return 0.0


def _perform_kmeans_clustering(
    embeddings: np.ndarray,
    k: int,
    use_bisecting_fallback: bool = True
) -> np.ndarray:
    """Perform k-means clustering with optional BisectingKMeans fallback.
    
    Args:
        embeddings: L2-normalized embeddings array
        k: Number of clusters
        use_bisecting_fallback: Whether to use BisectingKMeans if standard fails
        
    Returns:
        Cluster labels array
    """
    # Try standard KMeans first
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Check if we got all k clusters
    unique_labels = set(labels)
    if len(unique_labels) == k:
        return labels
    
    # Some clusters are empty - try BisectingKMeans as fallback
    if use_bisecting_fallback:
        logger.info(f"KMeans produced {len(unique_labels)} clusters instead of {k}, trying BisectingKMeans")
        try:
            bisecting = BisectingKMeans(n_clusters=k, random_state=42)
            labels = bisecting.fit_predict(embeddings)
            return labels
        except Exception as e:
            logger.warning(f"BisectingKMeans failed: {e}")
    
    # Return original labels even if not all k clusters
    return labels


def _select_optimal_k(
    embeddings: np.ndarray,
    branching_factor: int,
    n_papers: int,
    debug_log: list | None = None
) -> int:
    """Select optimal k for k-means clustering using quality scoring.
    
    Tries k = 2, 3, ... up to branching_factor (but not exceeding n_papers)
    and selects the smallest k whose score is close to the best score.
    
    Args:
        embeddings: L2-normalized embeddings array
        branching_factor: Maximum children before branching
        n_papers: Number of papers
        debug_log: Optional list to append debug info for analysis
        
    Returns:
        Optimal k value
    """
    # Determine candidate k values
    max_k = min(branching_factor, n_papers)
    candidate_ks = [k for k in range(2, max_k + 1)]
    
    if len(candidate_ks) == 0:
        if debug_log is not None:
            debug_log.append({"n_papers": n_papers, "reason": "no_candidates", "selected_k": 2})
        return 2
    
    if len(candidate_ks) == 1:
        if debug_log is not None:
            debug_log.append({"n_papers": n_papers, "reason": "single_candidate", "selected_k": candidate_ks[0]})
        return candidate_ks[0]
    
    # Try each k and compute quality score
    k_scores = {}
    for k in candidate_ks:
        if not _check_embeddings_valid(embeddings, k):
            continue
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Compute silhouette score
        score = _compute_silhouette_score(embeddings, labels)
        k_scores[k] = score
    
    if not k_scores:
        if debug_log is not None:
            debug_log.append({"n_papers": n_papers, "reason": "no_valid_k", "selected_k": 2})
        return 2
    
    # Find best score - use the k with highest silhouette score directly
    best_k = max(k_scores, key=k_scores.get)
    best_score = k_scores[best_k]
    selected_k = best_k  # Use best k directly, no tolerance-based preference for smaller k
    
    # Log debug info
    if debug_log is not None:
        debug_log.append({
            "n_papers": n_papers,
            "branching_factor": branching_factor,
            "k_scores": {k: round(v, 4) for k, v in k_scores.items()},
            "selected_k": selected_k,
            "best_score": round(best_score, 4),
        })
    
    return selected_k


# =============================================================================
# Tree Builder Class
# =============================================================================

class TreeBuilder:
    """Builds hierarchical tree directly in frontend-compatible format.
    
    The tree is built recursively, with each node returned as a dict:
    {
        "name": "node_xxx" (category) or "paper_<id>" (paper) - placeholder names,
        "node_id": "node_xxx",
        "node_type": "category" or "paper",
        "children": [...],  # for category nodes
        "paper_id": int     # for paper nodes
    }
    
    Single-child intermediate nodes are automatically unwrapped during building.
    """
    
    def __init__(self, embeddings_dict: dict[int, np.ndarray], branching_factor: int, debug_mode: bool = False):
        """Initialize tree builder.
        
        Args:
            embeddings_dict: Mapping of paper_id (int) to L2-normalized embedding vector
            branching_factor: Maximum children before a node should be split
            debug_mode: If True, collect clustering debug info
        """
        self.embeddings_dict = embeddings_dict
        self.branching_factor = branching_factor
        self.debug_mode = debug_mode
        self.debug_log: list = []  # Stores k-selection decisions
    
    def build_tree(self, paper_ids: list[int]) -> dict[str, Any]:
        """Build tree structure starting with given paper IDs.
        
        Args:
            paper_ids: List of integer paper IDs
            
        Returns:
            Frontend-compatible tree structure: {name, node_id, children: [...]}
        """
        if len(paper_ids) == 0:
            return {"name": "AI Papers", "node_id": "root", "node_type": "category", "children": []}
        
        if len(paper_ids) == 1:
            # Single paper - return as root with one child (wrapped in category for homogeneity)
            paper_node = self._create_paper_node(paper_ids[0])
            leaf_category = {
                "name": "Papers",
                "node_id": _generate_node_id(paper_ids),
                "node_type": "category",
                "children": [paper_node],
            }
            return {"name": "AI Papers", "node_id": "root", "node_type": "category", "children": [leaf_category]}
        
        # Build tree recursively
        root_node = self._build_node(paper_ids)
        
        # Wrap in root container
        if root_node.get("node_type") == "category" and root_node.get("children"):
            return {
                "name": "AI Papers",
                "node_id": "root",
                "node_type": "category",
                "children": root_node["children"],
            }
        else:
            return {
                "name": "AI Papers",
                "node_id": "root",
                "node_type": "category",
                "children": [root_node],
            }
    
    def _build_node(self, paper_ids: list[int]) -> dict[str, Any]:
        """Recursively build a node and its children.
        
        Args:
            paper_ids: List of integer paper IDs in this node
            
        Returns:
            Node dict in frontend format
        """
        node_id = _generate_node_id(paper_ids)
        
        # Base case: single paper
        if len(paper_ids) == 1:
            return self._create_paper_node(paper_ids[0])
        
        # Base case: small enough to be a leaf cluster
        if len(paper_ids) <= self.branching_factor:
            return self._create_leaf_cluster(paper_ids, node_id)
        
        # Need to split: get embeddings and cluster
        embeddings_list = [self.embeddings_dict[pid] for pid in paper_ids]
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        
        # Select optimal k (with optional debug logging)
        debug_log = self.debug_log if self.debug_mode else None
        k = _select_optimal_k(embeddings_array, self.branching_factor, len(paper_ids), debug_log)
        
        # Validate embeddings
        if not _check_embeddings_valid(embeddings_array, k):
            # Can't cluster - return as leaf
            return self._create_leaf_cluster(paper_ids, node_id)
        
        # Perform clustering with fallback
        cluster_labels = _perform_kmeans_clustering(embeddings_array, k)
        
        # Group papers by cluster label
        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(paper_ids[idx])
        
        # Filter empty clusters
        clusters = {label: papers for label, papers in clusters.items() if len(papers) > 0}
        
        # If clustering degenerated to single cluster, make it a leaf
        if len(clusters) <= 1:
            logger.info(f"Clustering degenerated to {len(clusters)} cluster(s) for {len(paper_ids)} papers, creating leaf")
            return self._create_leaf_cluster(paper_ids, node_id)
        
        # Build children recursively
        children = []
        for label in sorted(clusters.keys()):
            cluster_papers = clusters[label]
            child_node = self._build_node(cluster_papers)
            children.append(child_node)
        
        # Unwrap single-child intermediate: if only one child, return it directly
        if len(children) == 1:
            return children[0]
        
        # Ensure homogeneous children: all categories OR all papers, not mixed
        children = self._ensure_homogeneous_children(children)
        
        # Multiple children - create intermediate category node
        return {
            "name": node_id,  # Placeholder: use node_id verbatim, naming process will update
            "node_id": node_id,
            "node_type": "category",
            "children": children,
        }
    
    def _create_paper_node(self, paper_id: int) -> dict[str, Any]:
        """Create a paper node.
        
        Args:
            paper_id: Paper ID
            
        Returns:
            Paper node dict
        """
        node_id = _generate_node_id([paper_id])
        return {
            "name": f"paper_{paper_id}",  # Placeholder: paper_<id>, enrichment will add title
            "node_id": node_id,
            "node_type": "paper",
            "paper_id": paper_id,
        }
    
    def _create_leaf_cluster(self, paper_ids: list[int], node_id: str) -> dict[str, Any]:
        """Create a leaf cluster containing multiple papers.
        
        Args:
            paper_ids: List of paper IDs
            node_id: Node ID for this cluster
            
        Returns:
            Category node dict with paper children
        """
        children = [self._create_paper_node(pid) for pid in paper_ids]
        return {
            "name": node_id,  # Placeholder: use node_id verbatim, naming process will update
            "node_id": node_id,
            "node_type": "category",
            "children": children,
        }
    
    def _ensure_homogeneous_children(self, children: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ensure all children are the same type (all categories OR all papers).
        
        If children are mixed (some papers, some categories), wrap the papers
        in a "Miscellaneous" leaf category so all children become categories.
        
        Args:
            children: List of child nodes (can be mixed types)
            
        Returns:
            List of homogeneous children (all same node_type)
        """
        if not children:
            return children
        
        # Separate by type
        papers = [c for c in children if c.get("node_type") == "paper"]
        categories = [c for c in children if c.get("node_type") == "category"]
        
        # If all same type, no change needed
        if len(papers) == 0 or len(categories) == 0:
            return children
        
        # Mixed types: wrap papers in a "Miscellaneous" category
        # Collect paper_ids for generating a unique node_id
        # Add a prefix marker to distinguish from paper nodes with same paper_ids
        paper_ids = [p.get("paper_id") for p in papers if p.get("paper_id")]
        # Use negative IDs or a special marker to ensure uniqueness from paper nodes
        wrapper_ids = [-pid for pid in paper_ids] if paper_ids else [hash("misc")]
        misc_node_id = _generate_node_id(wrapper_ids)
        
        misc_category = {
            "name": misc_node_id,  # Placeholder: use node_id verbatim, naming process will update
            "node_id": misc_node_id,
            "node_type": "category",
            "children": papers,
        }
        
        # Return categories + the new misc category
        return categories + [misc_category]


# =============================================================================
# Database Operations
# =============================================================================

def _compute_category_centroids(
    tree: dict[str, Any],
    embeddings_dict: dict[int, np.ndarray],
) -> dict[str, tuple[np.ndarray, int]]:
    """Recursively compute mean centroid for every category node in the tree.

    Returns:
        Mapping of node_id → (centroid_vector, paper_count).
        Only category nodes are included (paper nodes are skipped).
    """
    def collect_paper_ids(node: dict[str, Any]) -> list[int]:
        if node.get("node_type") == "paper":
            pid = node.get("paper_id")
            return [pid] if pid is not None else []
        result = []
        for child in node.get("children", []):
            result.extend(collect_paper_ids(child))
        return result

    def walk(node: dict[str, Any], out: dict[str, tuple[np.ndarray, int]]) -> None:
        if node.get("node_type") != "category":
            return
        node_id = node.get("node_id")
        if node_id and node_id != "root":
            paper_ids = collect_paper_ids(node)
            valid = [embeddings_dict[pid] for pid in paper_ids if pid in embeddings_dict]
            if valid:
                centroid = np.mean(np.array(valid, dtype=np.float32), axis=0)
                # Re-normalise so cosine dot product comparisons stay in [−1, 1]
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                out[node_id] = (centroid, len(valid))
        for child in node.get("children", []):
            walk(child, out)

    result: dict[str, tuple[np.ndarray, int]] = {}
    walk(tree, result)
    return result


def write_tree_to_database(tree_structure: dict[str, Any], node_names: dict[str, str] | None = None) -> int:
    """Write frontend format tree structure to database as JSONB.
    
    Args:
        tree_structure: Frontend format tree structure {name, children: [...]}
        node_names: Optional dictionary mapping node_id to node name
    
    Returns:
        Number of nodes in tree
    """
    # Extract node_names from tree if not provided
    if node_names is None:
        node_names = {}
        def extract_names(node: dict[str, Any]):
            if node.get("node_id"):
                node_names[node["node_id"]] = node.get("name", "")
            if node.get("children"):
                for child in node["children"]:
                    extract_names(child)
        extract_names(tree_structure)
    
    # Count nodes
    def count_nodes(node: dict[str, Any]) -> int:
        count = 1
        if node.get("children"):
            for child in node["children"]:
                count += count_nodes(child)
        return count
    
    total_nodes = count_nodes(tree_structure)
    
    # Save to database
    db.save_tree(tree_structure, node_names)
    
    return total_nodes


# =============================================================================
# Main API
# =============================================================================

def _extract_papers_with_embeddings() -> tuple[list[dict], np.ndarray, list[int]]:
    """Extract papers with embeddings from database and L2-normalize them.
    
    Returns:
        Tuple of:
        - List of paper dictionaries
        - L2-normalized embeddings array (n_papers, embedding_dim)
        - List of paper IDs
    """
    papers = db.get_all_papers()
    
    papers_with_embeddings = []
    embeddings_list = []
    paper_ids_list = []
    
    for paper in papers:
        emb = paper.get("embedding")
        if emb is not None:
            papers_with_embeddings.append(paper)
            embeddings_list.append(_convert_embedding_to_numpy(emb))
            paper_ids_list.append(paper["id"])
    
    if len(embeddings_list) == 0:
        return [], np.array([]), []
    
    # Stack and L2-normalize embeddings
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    embeddings_normalized = normalize(embeddings_array, norm='l2')
    
    return papers_with_embeddings, embeddings_normalized, paper_ids_list


def build_tree_from_clusters() -> dict[str, Any]:
    """Build tree structure from all papers using hierarchical clustering.
    
    This is the main entry point for clustering. It:
    1. Fetches papers with embeddings from database
    2. L2-normalizes embeddings for cosine-similarity clustering
    3. Uses TreeBuilder to create frontend-format tree directly
    4. Returns tree structure ready for database storage
    
    Returns:
        Dictionary with tree structure and metadata:
        {
            "name": "AI Papers",
            "children": [...],
            "total_papers": int,
            "total_clusters": int
        }
    """
    # Extract and normalize embeddings
    papers_with_embeddings, embeddings_normalized, paper_ids_list = _extract_papers_with_embeddings()
    
    if len(papers_with_embeddings) < 2:
        return {
            "name": "AI Papers",
            "children": [],
            "total_papers": len(papers_with_embeddings),
            "total_clusters": 0,
            "message": f"Need at least 2 papers with embeddings, found {len(papers_with_embeddings)}",
        }
    
    # Create embeddings dictionary with normalized vectors
    embeddings_dict = {
        paper_id: embeddings_normalized[idx]
        for idx, paper_id in enumerate(paper_ids_list)
    }
    
    # Get clustering config
    config = _get_clustering_config()
    branching_factor = config["branching_factor"]
    
    # Build tree directly in frontend format (with debug mode enabled)
    builder = TreeBuilder(embeddings_dict, branching_factor, debug_mode=True)
    tree_structure = builder.build_tree(paper_ids_list)
    
    # Log debug info (visible in container logs)
    print(f"=== CLUSTERING DEBUG: branching_factor={branching_factor}, total_papers={len(papers_with_embeddings)} ===", flush=True)
    for decision in builder.debug_log:
        print(f"K-selection: {json.dumps(decision, default=lambda o: float(o) if hasattr(o, 'item') else str(o))}", flush=True)
    
    # Count total nodes in tree
    def count_nodes(node: dict[str, Any]) -> int:
        count = 1
        if node.get("children"):
            for child in node["children"]:
                count += count_nodes(child)
        return count
    
    total_nodes = count_nodes(tree_structure)
    
    # Compute and persist category centroids for incremental placement
    centroids = _compute_category_centroids(tree_structure, embeddings_dict)
    db.clear_category_embeddings()
    for node_id, (centroid, paper_count) in centroids.items():
        db.upsert_category_embedding(node_id, centroid.tolist(), paper_count)
    logger.info(f"Saved {len(centroids)} category centroids to DB")

    return {
        **tree_structure,
        "total_papers": len(papers_with_embeddings),
        "total_clusters": total_nodes,
    }


# =============================================================================
# Incremental Placement
# =============================================================================

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b))


def place_paper_in_tree(paper_id: int) -> dict[str, Any]:
    """Place a newly ingested paper into the existing tree using centroid descent.

    The caller (app.py) must hold ``_tree_lock`` before calling this function.

    Algorithm:
    1. Load paper embedding.
    2. Load tree + category_embeddings centroids from DB.
    3. Descend from root: at each level pick the child category with the
       highest cosine similarity to the paper embedding.
    4. Stop at the deepest category leaf reached.
    5. Append a new paper node to that leaf's children list.
    6. Recompute centroid for the modified leaf (running average).
    7. Mark the leaf node as dirty.
    8. Persist tree + centroid update to DB.

    Returns a result dict with placement info.
    """
    # 1. Load paper embedding
    raw_emb = db.get_paper_embedding(paper_id)
    if raw_emb is None:
        return {"placed": False, "reason": f"paper {paper_id} has no embedding"}

    paper_emb = _convert_embedding_to_numpy(raw_emb)
    # L2-normalise for cosine comparison
    norm = np.linalg.norm(paper_emb)
    if norm > 0:
        paper_emb = paper_emb / norm

    # 2. Load tree + centroids
    tree = db.get_tree()
    centroids = db.get_category_embeddings()  # node_id → {embedding, paper_count}

    # Convert centroids to numpy
    centroid_vecs: dict[str, np.ndarray] = {}
    centroid_counts: dict[str, int] = {}
    for nid, data in centroids.items():
        raw = data["embedding"]
        vec = _convert_embedding_to_numpy(raw)
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        centroid_vecs[nid] = vec
        centroid_counts[nid] = data["paper_count"]

    # 3–4. Descend from root to the best leaf category
    def find_best_leaf(node: dict[str, Any]) -> dict[str, Any]:
        """Return the deepest category node that best fits this paper."""
        if node.get("node_type") == "paper":
            return node

        children = node.get("children", [])
        category_children = [c for c in children if c.get("node_type") == "category"]

        if not category_children:
            # Leaf category (children are papers, or no children)
            return node

        # Find the child category with highest cosine similarity
        best_child = None
        best_sim = -2.0
        for child in category_children:
            cid = child.get("node_id")
            if cid and cid in centroid_vecs:
                sim = _cosine_similarity(paper_emb, centroid_vecs[cid])
                if sim > best_sim:
                    best_sim = sim
                    best_child = child

        if best_child is None:
            # No centroid available for any child → stay at current node
            return node

        return find_best_leaf(best_child)

    target_node = find_best_leaf(tree)

    # If descent ended at root or at a paper node, default to root
    if target_node.get("node_id") == "root" or target_node.get("node_type") == "paper":
        target_node = tree

    target_node_id = target_node.get("node_id", "root")

    # 5. Append new paper node
    new_paper_node = {
        "name": f"paper_{paper_id}",
        "node_id": _generate_node_id([paper_id]),
        "node_type": "paper",
        "paper_id": paper_id,
    }
    if "children" not in target_node:
        target_node["children"] = []
    target_node["children"].append(new_paper_node)

    # 6. Update centroid (running average)
    old_count = centroid_counts.get(target_node_id, 0)
    if target_node_id in centroid_vecs and old_count > 0:
        old_centroid = centroid_vecs[target_node_id]
        new_centroid = (old_centroid * old_count + paper_emb) / (old_count + 1)
        n = np.linalg.norm(new_centroid)
        if n > 0:
            new_centroid = new_centroid / n
    else:
        new_centroid = paper_emb
    new_count = old_count + 1
    db.upsert_category_embedding(target_node_id, new_centroid.tolist(), new_count)

    # 7. Mark node dirty
    db.add_dirty_tree_node(target_node_id)

    # 8. Persist tree
    db.save_tree(tree)

    logger.info(f"Placed paper {paper_id} into node '{target_node_id}'")
    return {"placed": True, "node_id": target_node_id, "paper_id": paper_id}


# =============================================================================
# Partial Recategorization
# =============================================================================

def _collect_paper_ids_under(node: dict[str, Any]) -> list[int]:
    """Recursively collect all paper_ids under a tree node."""
    if node.get("node_type") == "paper":
        pid = node.get("paper_id")
        return [pid] if pid is not None else []
    result = []
    for child in node.get("children", []):
        result.extend(_collect_paper_ids_under(child))
    return result


def _find_node_and_parent(
    tree: dict[str, Any],
    target_id: str,
    parent: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Find a node and its parent by node_id. Returns (node, parent) or (None, None)."""
    node_id = tree.get("node_id")
    if node_id == target_id:
        return tree, parent
    for child in tree.get("children", []):
        found, found_parent = _find_node_and_parent(child, target_id, tree)
        if found is not None:
            return found, found_parent
    return None, None


def partial_recategorize(dirty_node_ids: list[str]) -> dict[str, Any]:
    """Re-cluster the subtrees rooted at the given dirty nodes.

    The caller (app.py) must hold ``_tree_lock`` before calling this function.

    For each dirty node:
    1. Collect all paper_ids under that node.
    2. Re-run TreeBuilder._build_node() to produce a new sub-tree.
    3. Apply _ensure_homogeneous_children().
    4. Preserve the dirty node's existing node_id and name.
    5. Splice the new sub-tree's children back under the dirty node.
    6. Recompute centroids for all affected nodes.
    7. Clear dirty flags for processed nodes.
    8. Save tree.

    Returns a result dict with stats.
    """
    if not dirty_node_ids:
        return {"recategorized": 0, "nodes": []}

    # Load full embeddings for all papers (needed for re-clustering)
    _, embeddings_normalized, paper_ids_list = _extract_papers_with_embeddings()
    if len(embeddings_normalized) == 0:
        return {"recategorized": 0, "nodes": [], "message": "No papers with embeddings"}

    embeddings_dict = {
        paper_id: embeddings_normalized[idx]
        for idx, paper_id in enumerate(paper_ids_list)
    }

    config = _get_clustering_config()
    branching_factor = config["branching_factor"]
    builder = TreeBuilder(embeddings_dict, branching_factor)

    tree = db.get_tree()
    processed = []
    new_child_node_ids: list[str] = []

    for dirty_id in dirty_node_ids:
        node, parent = _find_node_and_parent(tree, dirty_id)
        if node is None:
            logger.warning(f"Dirty node {dirty_id} not found in tree, skipping")
            continue

        # 1. Collect paper_ids under this node
        paper_ids = _collect_paper_ids_under(node)
        if not paper_ids:
            logger.info(f"Dirty node {dirty_id} has no papers, skipping")
            continue

        # 2. Re-cluster
        valid_ids = [pid for pid in paper_ids if pid in embeddings_dict]
        if len(valid_ids) < 2:
            # Not enough to cluster — leave as-is, just clear dirty flag
            processed.append(dirty_id)
            continue

        new_subtree = builder._build_node(valid_ids)

        # 3. Apply homogeneous children
        if new_subtree.get("children"):
            new_subtree["children"] = builder._ensure_homogeneous_children(new_subtree["children"])

        # 4–5. Preserve existing node_id and name; replace children
        preserved_name = node.get("name", dirty_id)
        node["name"] = preserved_name
        # node["node_id"] stays as dirty_id (already in-place)

        if new_subtree.get("node_type") == "category" and new_subtree.get("children"):
            node["children"] = new_subtree["children"]
            # Collect ids of new child category nodes for naming
            for child in node["children"]:
                if child.get("node_type") == "category":
                    new_child_node_ids.append(child["node_id"])
        else:
            # Subtree collapsed to a leaf or paper — keep children as paper nodes
            node["children"] = [{"name": f"paper_{pid}", "node_id": _generate_node_id([pid]),
                                  "node_type": "paper", "paper_id": pid}
                                 for pid in valid_ids]

        processed.append(dirty_id)
        logger.info(f"Partial recategorize: rebuilt node {dirty_id} ({len(valid_ids)} papers)")

    # 6. Recompute all centroids from the updated tree
    centroids = _compute_category_centroids(tree, embeddings_dict)
    # Only update the affected nodes (the dirty ones and their new children)
    affected_ids = set(processed) | set(new_child_node_ids)
    for node_id, (centroid, paper_count) in centroids.items():
        if node_id in affected_ids:
            db.upsert_category_embedding(node_id, centroid.tolist(), paper_count)

    # 7. Clear dirty flags for processed nodes
    db.clear_dirty_tree_nodes(processed)

    # 8. Save tree
    db.save_tree(tree)

    return {
        "recategorized": len(processed),
        "nodes": processed,
        "new_child_nodes": new_child_node_ids,
    }


# =============================================================================
# Main Block - Standalone Execution
# =============================================================================

if __name__ == "__main__":
    """Sample calling script to run clustering as standalone function.
    
    Usage:
        python clustering.py
        
    This will:
        1. Load all papers with embeddings from the database
        2. L2-normalize embeddings
        3. Build a hierarchical clustering tree
        4. Print the tree structure as JSON
    """
    print("=" * 60)
    print("Running hierarchical clustering on all papers...")
    print("=" * 60)
    
    # Build tree from all papers
    result = build_tree_from_clusters()
    
    # Print summary
    print(f"\n✓ Clustering complete!")
    print(f"  - Total papers processed: {result.get('total_papers', 0)}")
    print(f"  - Total nodes in tree: {result.get('total_clusters', 0)}")
    
    if result.get('message'):
        print(f"  - Note: {result['message']}")
    
    # Print tree structure as JSON
    print("\n" + "=" * 60)
    print("Tree structure (saved to tree.json):")
    print("=" * 60)
    
    # Remove metadata for cleaner output (use storage dir for Singularity compatibility)
    import os
    os.makedirs("storage/schemas", exist_ok=True)
    output = {k: v for k, v in result.items() if k not in ['total_papers', 'total_clusters', 'message']}
    with open("storage/schemas/tree.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"Tree saved to storage/schemas/tree.json")
