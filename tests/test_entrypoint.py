"""Tests for the Docker entry point (docker/entrypoint.py)."""

from __future__ import annotations

import importlib.util
import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker" / "entrypoint.py"


@pytest.fixture(scope="module")
def entry():
    spec = importlib.util.spec_from_file_location("docker_entrypoint", ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_markdown_table_and_headings(entry):
    md = "# Title\n\n- item <b>\n\n| model | f1 |\n|---|---|\n| rf | 0.7 |\n"
    out = entry.markdown_to_html(md)
    assert "<h1>Title</h1>" in out
    assert "<li>item &lt;b&gt;</li>" in out  # escaped
    assert "<th>model</th>" in out and "<td>0.7</td>" in out
    assert "---" not in out  # the separator row is dropped


def test_cli_passthrough_and_index(entry, tmp_path, monkeypatch):
    monkeypatch.setenv("APP_VERSION", "12")
    monkeypatch.setenv("APP_COMMIT", "abc1234")
    assert entry.main(["demo", "--out-dir", str(tmp_path), "--n-students", "200"]) == 0
    page = entry.build_index(tmp_path).read_text(encoding="utf-8")
    assert "image build 12" in page and "commit abc1234" in page
    assert "<table>" in page and 'src="model_comparison.png"' in page


def test_healthz_and_static_files(entry, tmp_path):
    (tmp_path / "index.html").write_text("hello", encoding="utf-8")
    handler = partial(entry.ResultsHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/healthz", timeout=5) as r:
            assert r.status == 200 and r.read() == b"ok\n"
        with urllib.request.urlopen(f"{base}/", timeout=5) as r:
            assert r.read() == b"hello"
    finally:
        server.shutdown()
        server.server_close()
