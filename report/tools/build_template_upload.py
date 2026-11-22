"""Static checks for report/Elcvia2e.tex and build the upload folder + zip for the instructor's Overleaf template project."""
import re
import shutil
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
tex = (root / "report/Elcvia2e.tex").read_text(encoding="utf-8")
body = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())

opens = len(re.findall(r"(?<!\\)\{", body))
closes = len(re.findall(r"(?<!\\)\}", body))
print("unbalanced braces:", opens - closes)
begins = re.findall(r"\\begin\{(\w+\*?)\}", body)
ends = re.findall(r"\\end\{(\w+\*?)\}", body)
print("begin/end environments balanced:", sorted(begins) == sorted(ends))
labels = set(re.findall(r"\\label\{([^}]+)\}", body))
for f in (root / "results/tables").glob("*.tex"):
    labels |= set(re.findall(r"\\label\{([^}]+)\}", f.read_text(encoding="utf-8")))
refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
print("refs without label:", sorted(refs - labels))
bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
cites = {c.strip() for m in re.findall(r"\\cite\{([^}]+)\}", body) for c in m.split(",")}
print("cites without bibitem:", sorted(cites - bibitems), "| bibitems never cited:", sorted(bibitems - cites))
print("bare % in generated tables:", [f.name for f in (root / "results/tables").glob("*.tex")
                                        if re.search(r"(?<!\\)%", f.read_text(encoding="utf-8"))])

figs = sorted(set(re.findall(r"\\figdir/([\w.]+\.png)", tex)))
tabs = sorted(set(re.findall(r"\\tabdir/([\w.]+\.tex)", tex)))
assert figs and tabs, "no figures or tables found in the tex file"
print("figures:", figs)
print("tables:", tabs)

out = Path(sys.argv[1])
if out.exists():
    shutil.rmtree(out)
(out / "figures").mkdir(parents=True)
(out / "tables").mkdir()
shutil.copy(root / "report/Elcvia2e.tex", out / "Elcvia2e.tex")
for f in figs:
    shutil.copy(root / "results/figures" / f, out / "figures" / f)
for t in tabs:
    shutil.copy(root / "results/tables" / t, out / "tables" / t)
zpath = out.with_suffix(".zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(out).as_posix())
print("upload folder:", out)
print("zip:", zpath, zpath.stat().st_size, "bytes")
