"""
Unit tests for the LawUP frontend web dashboard and static file serving.
"""

import os
from fastapi.testclient import TestClient

from backend.app.api.main import app

client = TestClient(app)


def test_health_endpoint_not_shadowed_by_static_mount():
    """Verify API endpoints still function cleanly with static files mounted."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "lm_studio_url" in data


def test_dashboard_index_html_served():
    """Verify GET / returns index.html for the web dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "LawUP" in response.text
    assert "Contract Risk Analysis" in response.text
    assert "tab-document" in response.text


def test_dashboard_static_assets_served():
    """Verify static JS and CSS assets are served with proper headers."""
    css_res = client.get("/src/styles.css")
    assert css_res.status_code == 200
    assert "LawUP Modern Web Interface Styles" in css_res.text

    js_res = client.get("/src/app.js")
    assert js_res.status_code == 200
    assert "LawUP Web Application Logic" in js_res.text


def test_dashboard_package_json_exists():
    """Verify package.json exists in frontend directory."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pkg_path = os.path.join(project_root, "frontend", "package.json")
    assert os.path.isfile(pkg_path)
