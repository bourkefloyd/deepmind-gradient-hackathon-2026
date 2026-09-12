#!/usr/bin/env python3
"""Render nano/lab_notebook.ipynb to static/lab/index.html with Word Hunt VS theme."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "nano" / "lab_notebook.ipynb"
OUT = ROOT / "static" / "lab" / "index.html"
FALLBACK = ROOT / "nano" / "lab_notebook.html"
EXEC_TIMEOUT = 120

THEME_CSS = """
<style id="wh-lab-theme">
:root{
  --bg:#070b16;--panel:#0d1424;--panel2:#131c31;--line:#1c2a47;
  --ink:#eef2fb;--muted:#8391ad;--accent:#2ee87a;--accent-ink:#052b14;--blue:#3b9dff;
  --jp-ui-font-color1:var(--ink);--jp-ui-font-color2:var(--muted);--jp-ui-font-color3:var(--muted);
  --jp-content-font-color1:var(--ink);--jp-content-font-color2:var(--muted);--jp-content-font-color3:var(--muted);
  --jp-layout-color0:var(--bg);--jp-layout-color1:var(--panel);--jp-layout-color2:var(--panel2);--jp-layout-color3:var(--line);
  --jp-border-color1:var(--line);--jp-border-color2:var(--line);
  --jp-cell-editor-background:var(--panel2);--jp-cell-editor-active-background:var(--panel);
  --jp-rendermime-table-row-background:var(--panel);--jp-rendermime-table-row-hover-background:var(--panel2);
  --jp-mirror-editor-variable-color:var(--ink);--jp-mirror-editor-comment-color:var(--muted);
  --jp-mirror-editor-keyword-color:var(--blue);--jp-mirror-editor-string-color:var(--accent);
  --jp-mirror-editor-number-color:#f5b301;--jp-mirror-editor-operator-color:var(--blue);
  --jp-mirror-editor-punctuation-color:var(--muted);--jp-mirror-editor-error-color:#ff5d6c;
}
html,body{background:radial-gradient(1200px 600px at 50% -10%,#0f1a33 0%,var(--bg) 60%) fixed,var(--bg)!important;color:var(--ink)!important;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
body.jp-Notebook{padding-top:0}
.wh-lab-header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 24px;background:rgba(13,20,36,.92);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:100;backdrop-filter:blur(8px)}
.wh-lab-brand{display:flex;align-items:center;gap:12px}
.wh-lab-logo{width:40px;height:40px;border-radius:10px;background:var(--accent);display:grid;place-items:center;box-shadow:0 0 18px rgba(46,232,122,.35)}
.wh-lab-logo svg{width:24px;height:24px}
.wh-lab-brand b{font-size:20px;font-weight:900;color:var(--ink)}
.wh-lab-brand em{font-style:normal;color:var(--accent)}
.wh-lab-back{color:var(--muted);text-decoration:none;font-weight:700;font-size:14px;padding:10px 14px;border:1px solid var(--line);border-radius:12px;background:var(--panel)}
.wh-lab-back:hover{color:var(--ink);border-color:var(--accent)}
main.jp-Notebook,main{max-width:960px;margin:0 auto;padding:24px 20px 48px}
.jp-RenderedHTMLCommon table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
.jp-RenderedHTMLCommon th,.jp-RenderedHTMLCommon td{border:1px solid var(--line);padding:8px 10px;text-align:left}
.jp-RenderedHTMLCommon th{background:var(--panel2);color:var(--ink)}
.jp-RenderedHTMLCommon tr:nth-child(even){background:rgba(19,28,49,.5)}
.jp-RenderedHTMLCommon h1,.jp-RenderedHTMLCommon h2,.jp-RenderedHTMLCommon h3{color:var(--ink)}
.jp-RenderedHTMLCommon code{background:var(--panel2);padding:2px 6px;border-radius:6px;color:var(--blue)}
.jp-RenderedHTMLCommon pre{background:var(--panel2)!important;border:1px solid var(--line);border-radius:12px;padding:14px;overflow:auto}
.jp-RenderedHTMLCommon a.anchor-link{display:none}
.jp-Cell{padding:8px 0}
.jp-Collapser,.jp-InputPrompt{display:none!important}
.jp-InputArea{padding-left:0!important}
.jp-OutputArea-output img,.jp-RenderedImage img{max-width:100%;background:#fff;border-radius:8px}
</style>
"""

HEADER = """
<header class="wh-lab-header">
  <div class="wh-lab-brand">
    <span class="wh-lab-logo"><svg viewBox="0 0 24 24" fill="none" stroke="#052b14" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7.5" height="7.5" rx="2"/><rect x="13.5" y="3" width="7.5" height="7.5" rx="2"/><rect x="3" y="13.5" width="7.5" height="7.5" rx="2"/><rect x="13.5" y="13.5" width="7.5" height="7.5" rx="2"/></svg></span>
    <div><b>Word Hunt <em>VS</em></b></div>
  </div>
  <a class="wh-lab-back" href="/">← Back to game</a>
</header>
"""


def apply_theme(html: str) -> str:
    html = re.sub(r"<title>[^<]*</title>", "<title>Word Hunt VS — Lab (Deep Dive)</title>", html, count=1)
    html = re.sub(
        r'data-jp-theme-light="[^"]*"\s*data-jp-theme-name="[^"]*"',
        'data-jp-theme-light="false" data-jp-theme-name="Word Hunt VS"',
        html,
        count=1,
    )
    if "wh-lab-theme" not in html:
        html = html.replace("</head>", THEME_CSS + "\n</head>")
    if "wh-lab-header" not in html:
        html = re.sub(
            r"(<body[^>]*>)",
            r"\1" + HEADER,
            html,
            count=1,
        )
    return html


def nbconvert(execute: bool) -> bool:
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "html",
        "--output", "index",
        "--output-dir", str(OUT.parent),
        str(NB),
    ]
    if execute:
        cmd.insert(4, "--execute")
        cmd.insert(5, "--ExecutePreprocessor.timeout=110")
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, timeout=EXEC_TIMEOUT if execute else 300)
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"nbconvert failed: {e}", file=sys.stderr)
        return False


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mode = "execute"
    if not nbconvert(execute=True):
        print("execute timed out or failed; trying without execute", file=sys.stderr)
        mode = "no-execute"
        if not nbconvert(execute=False):
            if not FALLBACK.is_file():
                sys.exit("nbconvert failed and no fallback html")
            print(f"using fallback {FALLBACK}", file=sys.stderr)
            html = FALLBACK.read_text(encoding="utf-8")
            mode = "fallback"
        else:
            html = OUT.read_text(encoding="utf-8")
    else:
        html = OUT.read_text(encoding="utf-8")
    OUT.write_text(apply_theme(html), encoding="utf-8")
    print(f"rendered {OUT} ({mode}, {OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
