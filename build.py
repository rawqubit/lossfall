#!/usr/bin/env python3
"""Inline engine.js, app.js and corpus.txt into index.html.

index.html is what ships; it must work from a file:// path with no network,
so everything is folded into one document. Run `python3 build.py` after editing
any source, or `python3 build.py --check` to verify index.html is in sync.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build() -> str:
    shell = (ROOT / "shell.html").read_text(encoding="utf-8")
    engine = (ROOT / "engine.js").read_text(encoding="utf-8")
    app = (ROOT / "app.js").read_text(encoding="utf-8")
    corpus = (ROOT / "corpus.txt").read_text(encoding="utf-8")

    for name, source in (("engine.js", engine), ("app.js", app)):
        if "</script" in source:
            raise SystemExit(f"build: {name} contains a script terminator and cannot be inlined")

    shell = shell.replace(
        "<!--ENGINE-->",
        "<script>\n/* ==ENGINE-START== */\n" + engine + "/* ==ENGINE-END== */\n</script>",
    )
    shell = shell.replace("__CORPUS__", json.dumps(corpus))
    shell = shell.replace("<!--APP-->", "<script>\n" + app + "</script>")
    return shell


def main() -> int:
    target = ROOT / "index.html"
    built = build()
    if "--check" in sys.argv:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != built:
            print("build: index.html is out of date, run python3 build.py", file=sys.stderr)
            return 1
        print(f"build: index.html is in sync ({len(built):,} bytes)")
        return 0
    target.write_text(built, encoding="utf-8")
    print(f"build: wrote index.html ({len(built):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
