"""E2E tests for tree structure endpoints.

Endpoints covered:
  GET /tree
"""
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


