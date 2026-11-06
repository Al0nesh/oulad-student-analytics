"""Entry point of the Docker image.

    docker run IMAGE <arguments>                 runs the normal CLI (demo, train, features, ...)
    docker run -p 8000:8000 IMAGE serve [args]   runs the demo, then serves the results over HTTP

The ``serve`` mode turns the batch pipeline into a small web service, so the same image
can be deployed to a test server or to Kubernetes and checked in a browser. ``/healthz``
answers ``ok`` for health checks and Kubernetes probes.
"""

from __future__ import annotations

import html
import os
import signal
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from oulad_analytics import __version__
from oulad_analytics.cli import main as cli_main

FIGURES = [
    ("model_comparison.png", "F1, recall and ROC-AUC of every model (test set)"),
    ("weekly_clicks.png", "Median weekly VLE activity: at-risk vs. successful students"),
    ("feature_importance.png", "Features the random forest relies on"),
]


def markdown_to_html(text: str) -> str:
    """Convert the small subset of Markdown used in ``summary.md`` to HTML."""
    out: list[str] = []
    table: list[list[str]] = []

    def flush_table() -> None:
        if not table:
            return
        head, *rows = table
        out.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>")
        out.extend("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        out.append("</table>")
        table.clear()

    for raw in text.splitlines():
        line = html.escape(raw.strip()).replace("`", "")
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(set(c) <= {"-", ":"} for c in cells):
                table.append(cells)
            continue
        flush_table()
        if line.startswith("## "):
            out.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- "):
            out.append(f"<li>{line[2:]}</li>")
        elif line:
            out.append(f"<p>{line}</p>")
    flush_table()
    return "\n".join(out)


def build_index(out_dir: Path) -> Path:
    """Write ``index.html`` with the summary table, the figures and the build information."""
    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    figures = "\n".join(
        f'<figure><img src="{name}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
        for name, caption in FIGURES
        if (out_dir / name).exists()
    )
    build = (
        f"oulad-analytics {__version__} &middot; image build {os.environ.get('APP_VERSION', 'dev')}"
        f" &middot; commit {os.environ.get('APP_COMMIT', 'unknown')}"
        f" &middot; host {html.escape(os.environ.get('HOSTNAME', 'local'))}"
    )
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>oulad-analytics demo</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem;
       color: #1f2933; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #cbd2d9; padding: .35rem .7rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f0f4f8; }}
img {{ max-width: 100%; border: 1px solid #e4e7eb; }}
figure {{ margin: 1.5rem 0; }}
.build {{ background: #e8f5e9; padding: .5rem .8rem; border-radius: 6px; font-size: .9rem; }}
</style></head><body>
<p class="build">{build}</p>
{markdown_to_html(summary)}
<p>Synthetic demo data: these numbers only show that the deployed pipeline works.</p>
{figures}
<p><a href="metrics.csv">metrics.csv</a> &middot;
<a href="feature_importance.csv">feature_importance.csv</a> &middot;
<a href="summary.md">summary.md</a></p>
</body></html>
"""
    index = out_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


class ResultsHandler(SimpleHTTPRequestHandler):
    """Static file server for the results folder plus a ``/healthz`` endpoint."""

    def do_GET(self) -> None:  # noqa: N802 - name fixed by http.server
        if self.path == "/healthz":
            body = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def serve(args: list[str]) -> int:
    out_dir = Path(os.environ.get("RESULTS_DIR", "/results"))
    port = int(os.environ.get("PORT", "8000"))
    code = cli_main(["demo", "--out-dir", str(out_dir), *args])
    if code:
        return code
    build_index(out_dir)
    handler = partial(ResultsHandler, directory=str(out_dir))
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    # As PID 1 in a container, Python ignores SIGTERM unless told otherwise; exit cleanly
    # so `docker stop` and Kubernetes rolling updates do not wait for the kill timeout.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"Serving results on http://0.0.0.0:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "serve":
        return serve(argv[1:])
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
