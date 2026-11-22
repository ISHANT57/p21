"""Build report/Elcvia2e.tex: the report written in the structure of the professor's ELCVIA template (elcvia2e.tex).

The file uses the real class files of the template (\\usepackage{elcvia}, psfig.sty), which are NOT in this
repository: put Elcvia2e.tex, the figures/ folder and the tables/ folder into a copy of the template project on
Overleaf and compile there. The body text is taken from report/main.tex (the stand-alone version), so both versions
say the same. The template has a hand-written reference list (thebibliography), so references.bib is converted to
that format here, in the order of first citation.
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
main = (root / "main.tex").read_text(encoding="utf-8")
bib_text = (root / "references.bib").read_text(encoding="utf-8")

abstract = re.search(r"\\begin\{list\}.*?\\item\[\] (.*?)\n\\end\{list\}", main, flags=re.S).group(1).strip()
keywords = re.search(r"\\textit\{Key Words\}: (.*?)\.(?:\\par)?\n", main).group(1)
body = main[main.index("\\section{Introduction}"): main.index("{\\footnotesize\n\\bibliographystyle{plain}")]

# ---- parse references.bib (small, regular format written by hand) -----------------------------------------------------------------
entries = {}
for m in re.finditer(r"@(\w+)\{([^,]+),\n(.*?)\n\}\n", bib_text + "\n", flags=re.S):
    kind, key, fields_text = m.group(1), m.group(2).strip(), m.group(3)
    fields = {}
    for fm in re.finditer(r"^\s*(\w+)\s*=\s*", fields_text, flags=re.M):
        i = fm.end()
        if fields_text[i] == "{":  # braced value, may contain nested braces
            depth, j = 0, i
            while True:
                depth += fields_text[j] == "{"
                depth -= fields_text[j] == "}"
                j += 1
                if depth == 0:
                    break
            fields[fm.group(1)] = fields_text[i + 1: j - 1]
        else:  # plain value such as a year
            fields[fm.group(1)] = re.match(r"[^,\n]+", fields_text[i:]).group(0).strip()
    entries[key] = (kind, fields)


def initials(given: str) -> str:
    """'Dong-Hyun' -> 'D.-H.'; 'Colin A.' -> 'C.A.'; the template writes initials without spaces."""
    out = []
    for token in given.split():
        parts = [p for p in token.split("-") if p]
        out.append("-".join((p.strip(".")[0] if not p.startswith("{") else p[:4]) + "." for p in parts))
    return "".join(out)


def authors(field: str) -> str:
    names = [n.strip() for n in field.split(" and ")]
    out = []
    for n in names:
        if n == "others":
            out.append("et al.")
            continue
        last, given = [p.strip() for p in n.split(",", 1)]
        out.append(f"{initials(given)} {last}")
    if out and out[-1] == "et al.":
        return ", ".join(out[:-1]) + " et al."
    return ", ".join(out[:-1]) + (" and " if len(out) > 1 else "") + out[-1]


def pages(p: str) -> str:
    return p.replace("--", "-")


def format_entry(kind: str, f: dict) -> str:
    title = f["title"]
    head = f"{authors(f['author'])}, ``{title},'' "
    if kind == "article":
        vol = f.get("volume", "")
        num = f"({f['number']})" if "number" in f else ""
        tail = f"{{\\em {f['journal']}}}" + (f" {vol}{num}" if vol else "") + (f":{pages(f['pages'])}" if "pages" in f else "") + f", {f['year']}."
    elif kind == "inproceedings":
        tail = f"{{\\em {f['booktitle']}}}" + (f", {pages(f['pages'])}" if "pages" in f else "") + f", {f['year']}."
    else:  # misc
        tail = f"{f['howpublished']}, {f['year']}."
    return head + tail


# ---- order of first citation -------------------------------------------------------------------------------------------------------
order = []
for m in re.finditer(r"\\cite\{([^}]+)\}", body):
    for key in m.group(1).split(","):
        key = key.strip()
        if key not in order:
            order.append(key)
missing = [k for k in order if k not in entries]
assert not missing, f"cited but not in references.bib: {missing}"
bibitems = "\n\n".join(f"\\bibitem{{{k}}}\n{format_entry(*entries[k])}" for k in order)

HEAD = r"""%**************************************************************************
%*  Label-Efficient Land-Use Classification on EuroSAT                    *
%*  Deep Learning course project, Team 21 (project P21)                   *
%*  Format: ELECTRONIC LETTERS ON COMPUTER VISION AND IMAGE ANALYSIS      *
%*          (LaTeX 2e template provided by the course instructor)         *
%**************************************************************************
% HOW TO COMPILE: this file needs the class files of the instructor's template (elcvia.sty, elcvia.cls, psfig.sty).
% Open a copy of the template project on Overleaf, upload this file (it replaces Elcvia2e.tex) together with the
% folders figures/ and tables/, and recompile. After compiling, set \finpag below to the real last page.

\documentclass[twoside,a4paper,11pt]{article}
\usepackage{times}
\usepackage{elcvia}
\usepackage[english]{babel}
% extra packages for the figures, the tables (booktabs rules), the formulas and the web addresses
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{array}
\usepackage{url}

\newcommand{\Runo}{\rm I\hspace{-0.55ex}R}

\input{psfig.sty}
%\psdraft

% folders of the figures and of the generated tables (next to this file on Overleaf)
\newcommand{\figdir}{figures}
\newcommand{\tabdir}{tables}

% INFORMATION TO FILL IN CAMERA READY VERSION
%------------------------------------------------------------------
\newcommand{\vol}{0}            % Number of Volume (default 0)
\newcommand{\num}{0}            % Number (default 0)
\setcounter{page}{1}            % First page (default 1)
\newcommand{\begpag}{\thepage}
\newcommand{\finpag}{12}        % Last page (update to the real last page after compiling)
\newcommand{\yearp}{2026}       % Year of the issue
%------------------------------------------------------------------
\renewcommand{\thefootnote}{\fnsymbol{footnote}}

\title{Label-Efficient Land-Use Classification on EuroSAT:
RGB versus 13-Band Input, Supervised versus Pseudo-Labelled Training
\footnote[0]{Deep Learning course project, Team 21 (project P21)}
% journal information of the template (no e-mail and no associate-editor name are given)
\footnote[0]{ELCVIA ISSN:1577-5097}
\footnote[0]{Published by Computer Vision Center / Universitat Aut\`onoma de
Barcelona, Barcelona, Spain}
}

\author{Ishant Bhoyar$^*$, Gaurav Kumar$^*$, Jagriti Pandey$^*$ and Ayush Kumar$^*$ \\ \\
\centerline{\small \em $^*$ Deep Learning, Sitare University (enrolment nos. 202410101200059, 202410101200047,
202410101200060 and 202410101200028)} \\ \\
% Received date (the revised and accepted dates of the template are filled in by the journal, so they are left out)
\centerline{\small Received 22 November 2026}
       }


\begin{document}

\maketitle

\pagestyle{myheadings}

% Running head of the template: authors in the format "J. Jones et al." for more than two authors.
\markboth{\centerline{\small \it Bhoyar et al. /
           Electronic Letters on Computer Vision and Image Analysis
           \vol(\num):\begpag-\finpag, \yearp}}
         {\centerline{\small \it Bhoyar et al. /
           Electronic Letters on Computer Vision and Image Analysis
           \vol(\num):\begpag-\finpag, \yearp}}
\hrulefill

\begin{abstract}

ABSTRACT_TEXT

\vspace{0.5cm}
\noindent
{\em Key Words}: KEYWORDS_TEXT.

\end{abstract}
\hrulefill

"""
head = HEAD.replace("ABSTRACT_TEXT", abstract).replace("KEYWORDS_TEXT", keywords)

tail = "\n\\begin{thebibliography}{1}\n\n" + bibitems + "\n\n\\end{thebibliography}\n\n\\end{document}\n"
out = head + body.rstrip() + "\n" + tail
assert "\\bibliography{" not in out and "IEEE" not in out.replace("IEEE Journal", "").replace("IEEE Geoscience", "").replace("IEEE Conf", "").replace("IEEE Int", "")
(root / "Elcvia2e.tex").write_text(out, encoding="utf-8")
print("written", len(out.splitlines()), "lines;", len(order), "references")
