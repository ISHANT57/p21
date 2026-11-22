"""Build a self-contained Overleaf upload from report/ and results/ (paths rewritten, repo untouched)."""
import re
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])

tex = (root / "report/main.tex").read_text(encoding="utf-8")
tex2 = tex.replace(r"\newcommand{\figdir}{../results/figures}", r"\newcommand{\figdir}{figures}")
tex2 = tex2.replace(r"\newcommand{\tabdir}{../results/tables}", r"\newcommand{\tabdir}{tables}")
assert tex2 != tex and "figdir}{figures}" in tex2 and "tabdir}{tables}" in tex2

used_figs = set(re.findall(r"\\figdir\}?/([\w.]+\.png)", tex2)) or set(re.findall(r"figdir/([\w.]+\.png)", tex2))
used_tabs = set(re.findall(r"tabdir/([\w.]+\.tex)", tex2))
print("figures:", sorted(used_figs))
print("tables:", sorted(used_tabs))

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("main.tex", tex2)
    z.write(root / "report/references.bib", "references.bib")
    for f in sorted(used_figs):
        z.write(root / "results/figures" / f, f"figures/{f}")
    for t in sorted(used_tabs):
        z.write(root / "results/tables" / t, f"tables/{t}")
with zipfile.ZipFile(out) as z:
    for i in z.infolist():
        print(f"{i.file_size:>9}  {i.filename}")
