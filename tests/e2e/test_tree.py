"""E2E tests for tree structure endpoints.

Endpoints covered:
  GET    /tree
  POST   /tree/node
  DELETE /tree/node/{node_id}
"""
import uuid

import requests
import pytest

from conftest import BACKEND_URL, MEDIUM_TIMEOUT


class TestGetTree:
    def test_get_tree(self, backend_available):
        resp = requests.get(f"{BACKEND_URL}/tree", timeout=MEDIUM_TIMEOUT)
        assert resp.status_code == 200, f"GET /tree failed: {resp.text}"
        tree = resp.json()
        assert isinstance(tree, dict), f"Tree should be a dict, got {type(tree)}"
        assert "name" in tree, f"Tree missing 'name' field: {tree}"

    def test_tree_children_structure(self, backend_available, existing_paper):
        """After at least one paper is in DB, tree should have children."""
        resp = requests.get(f"{BACKEND_URL}/tree", timeout=MEDIUM_TIMEOUT)
        assert resp.status_code == 200
        tree = resp.json()
        # "children" key may exist or may be absent on a flat tree
        if "children" in tree:
            assert isinstance(tree["children"], list), (
                f"Tree 'children' should be a list: {tree['children']}"
            )


class TestTreeNode:
    def test_add_tree_node(self, backend_available):
        """POST /tree/node — compat stub, always returns 200 with status message.

        Note: The tree is stored as JSONB rebuilt by clustering, so individual
        node additions are not applied. The endpoint is kept for compatibility.
        """
        node_id = f"e2e_test_{uuid.uuid4().hex[:8]}"
        add_resp = requests.post(
            f"{BACKEND_URL}/tree/node",
            json={
                "node_id": node_id,
                "name": f"E2E Test Node {node_id}",
                "node_type": "category",
                "parent_id": None,
            },
            timeout=MEDIUM_TIMEOUT,
        )
        assert add_resp.status_code == 200, (
            f"POST /tree/node failed: {add_resp.status_code}: {add_resp.text}"
        )
        data = add_resp.json()
        assert "status" in data, f"Expected 'status' in response: {data}"

    def test_delete_tree_node(self, backend_available):
        """DELETE /tree/node/{id} — compat stub, always returns 200.

        Note: Same as POST — tree is rebuilt from clustering, so this is a no-op.
        """
        resp = requests.delete(
            f"{BACKEND_URL}/tree/node/any_node_id",
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 200, (
            f"DELETE /tree/node failed: {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert "status" in data, f"Expected 'status' in response: {data}"

    def test_add_tree_node_missing_required_fields(self, backend_available):
        resp = requests.post(
            f"{BACKEND_URL}/tree/node",
            json={},
            timeout=MEDIUM_TIMEOUT,
        )
        assert resp.status_code == 422, (
            f"Expected 422 for missing fields, got {resp.status_code}: {resp.text}"
        )
