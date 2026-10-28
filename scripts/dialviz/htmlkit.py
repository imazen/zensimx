"""HTML helpers for dialviz: page shell, escaping, inline markdown, chips, tables.

Everything here formats values that the source readers (`sources.py`) already
extracted. Nothing computes a statistic.
"""
from __future__ import annotations

import html
import re

GITHUB_BLOB = "https://github.com/imazen/zensimx/blob/main/"

# Set by the build (gitref.Linker): maps a cited path and line to a GitHub permalink.
_LINKER = None


def set_linker(linker) -> None:
    global _LINKER
    _LINKER = linker


def blob_url(path: str, line: int | None = None) -> str:
    if _LINKER is not None:
        return _LINKER(path, line)
    if path.startswith("zenanalyze:"):
        return "https://github.com/imazen/zenanalyze/blob/main/" + path.split(":", 1)[1] + (f"#L{line}" if line else "")
    return GITHUB_BLOB + path + (f"#L{line}" if line else "")

NAV = [
    ("index.html", "Overview"),
    ("readiness.html", "Readiness"),
    ("properties.html", "Wanted properties"),
    ("gates.html", "Gates and defects"),
    ("evaluation.html", "Evaluation and verdicts"),
    ("experiments.html", "Experiments"),
    ("splits.html", "Data roles"),
    ("features.html", "zensim features"),
    ("zenanalyze.html", "zenanalyze features"),
    ("sources.html", "Sources"),
]


def esc(s) -> str:
    return html.escape("" if s is None else str(s), quote=True)


_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<![\w*])\*(?!\s)([^*]+?)\*(?![\w*])")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def repo_link(target: str, src_path: str | None) -> str:
    """Resolve a markdown link target found in repo file `src_path` to a URL."""
    if re.match(r"^[a-z]+://", target) or target.startswith("#"):
        return target
    if src_path is None:
        return target
    sp = src_path.split(":", 1)[1] if src_path.startswith("zenanalyze:") else src_path
    base = sp.rsplit("/", 1)[0] if "/" in sp else ""
    parts = (base + "/" + target).split("/") if base else target.split("/")
    out: list[str] = []
    for p in parts:
        if p in ("", "."):
            continue
        if p == "..":
            if out:
                out.pop()
            continue
        out.append(p)
    prefix = "zenanalyze:" if src_path.startswith("zenanalyze:") else ""
    return blob_url(prefix + "/".join(out))


def md_inline(text: str, src_path: str | None = None) -> str:
    """Render the inline markdown used in repo tables (code, bold, italics, links)."""
    if text is None:
        return ""
    stash: list[str] = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    t = _CODE.sub(lambda m: keep(f"<code>{esc(m.group(1))}</code>"), text)
    t = _LINK.sub(lambda m: keep(f'<a href="{esc(repo_link(m.group(2), src_path))}">{esc(m.group(1))}</a>'), t)
    t = esc(t)
    t = _BOLD.sub(r"<strong>\1</strong>", t)
    t = _ITAL.sub(r"<em>\1</em>", t)
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], t)


def md_block(text: str, src_path: str | None = None) -> str:
    """Paragraphs, bullet lists and GFM tables from a markdown excerpt."""
    from .mdparse import split_row
    out: list[str] = []
    para: list[str] = []
    items: list[str] = []
    trows: list[str] = []

    def flush_table():
        if not trows:
            return
        rows = [split_row(r) for r in trows if not re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?", r.strip())]
        head, body = rows[0], rows[1:]
        out.append('<div class="tablewrap"><table><thead><tr>' + "".join(f"<th>{md_inline(c, src_path)}</th>" for c in head)
                   + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{md_inline(c, src_path)}</td>" for c in r) + "</tr>"
                                                      for r in body) + "</tbody></table></div>")
        trows.clear()

    def flush():
        flush_table()
        if para:
            out.append("<p>" + md_inline(" ".join(para), src_path) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{md_inline(i, src_path)}</li>" for i in items) + "</ul>")
            items.clear()

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            if para or items:
                flush()
            trows.append(s)
            continue
        flush_table()
        m = re.match(r"^[-*]\s+(.*)", s)
        hm = re.match(r"^#{1,6}\s+(.*)", s)
        if hm:
            flush()
            out.append(f"<p><strong>{md_inline(hm.group(1), src_path)}</strong></p>")
        elif not s:
            flush()
        elif m:
            if para:
                flush()
            items.append(m.group(1))
        elif items and line.startswith("  "):
            items[-1] += " " + s
        else:
            if items:
                flush()
            para.append(s)
    flush()
    return "\n".join(out)


STATUS_LABEL = {
    "pass": "pass", "fail": "fail", "blocked": "blocked", "ready": "ready", "done": "done",
    "pending": "rule pending", "na": "not measured", "open": "open", "fixed": "fixed", "info": "info",
}


def chip(state: str, label: str | None = None, tip: str | None = None) -> str:
    st = state if state in STATUS_LABEL else "info"
    t = f' data-tip="{esc(tip)}"' if tip else ""
    return f'<span class="chip st-{st}"{t}>{esc(label or STATUS_LABEL[st])}</span>'


def src_cite(path: str, line: int | None = None, label: str | None = None) -> str:
    shown = label or (f"{path}:{line}" if line else path)
    return f'<a class="mono small" href="{esc(blob_url(path, line))}">{esc(shown)}</a>'


def table(headers, rows, *, tid: str | None = None, sortable: bool = True, numeric_cols=(), cls: str = "") -> str:
    """rows: list of lists of pre-rendered HTML cells (or (html, data_v) tuples)."""
    classes = " ".join(c for c in ("sortable" if sortable else "", cls) if c)
    idattr = f' id="{esc(tid)}"' if tid else ""
    head = "".join(f'<th class="num">{h}</th>' if i in numeric_cols else f"<th>{h}</th>" for i, h in enumerate(headers))
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            v = None
            if isinstance(c, tuple):
                c, v = c
            vattr = f' data-v="{esc(v)}"' if v is not None else ""
            klass = ' class="num"' if i in numeric_cols else ""
            cells.append(f"<td{klass}{vattr}>{c}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="tablewrap"><table class="{classes}"{idattr}><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def filterbar(tid: str, placeholder: str = "Filter rows", selects=()) -> str:
    """selects: iterable of (column_index, label, values)."""
    parts = [f'<input type="search" data-filter="{esc(tid)}" placeholder="{esc(placeholder)}" aria-label="{esc(placeholder)}">']
    for col, label, values in selects:
        opts = "".join(f'<option value="{esc(v.lower())}">{esc(v)}</option>' for v in values)
        parts.append(f'<select data-filter="{esc(tid)}" data-filter-col="{col}" aria-label="{esc(label)}">'
                     f'<option value="">{esc(label)}: all</option>{opts}</select>')
    parts.append(f'<span class="count" data-count="{esc(tid)}"></span>')
    return '<div class="filterbar">' + "".join(parts) + "</div>"


def page(*, title: str, current: str, body: str, depth: int = 0, crumbs=(), build_info: str = "") -> str:
    base = "../" * depth
    nav = "".join(
        f'<a href="{base}{href}"{" aria-current=\"page\"" if href == current else ""}>{esc(label)}</a>'
        for href, label in NAV
    )
    crumb = ""
    if crumbs:
        crumb = '<div class="crumbs">' + " / ".join(
            f'<a href="{base}{h}">{esc(t)}</a>' if h else esc(t) for t, h in crumbs) + "</div>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} · dialviz</title>
<link rel="stylesheet" href="{base}assets/style.css">
<script>try{{var t=localStorage.getItem("dialviz-theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}}catch(e){{}}</script>
</head>
<body data-base="{base}">
<header class="top"><div class="top-inner">
<a class="brand" href="{base}index.html">zensim dialviz</a>
<nav class="main" aria-label="Sections">{nav}</nav>
<div class="tools"><input id="q" type="search" placeholder="Search ( / )" aria-label="Search the site" autocomplete="off">
<button id="theme" type="button" aria-label="Toggle colour theme">Auto</button></div>
<div id="results" role="listbox"></div>
</div></header>
<main>{crumb}
{body}
</main>
<footer>{build_info}</footer>
<script src="{base}search.js"></script>
<script src="{base}assets/app.js"></script>
</body></html>
"""
