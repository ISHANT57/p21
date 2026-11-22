"""Build report/main.tex in the layout of the ELCVIA journal template from report/main_ieee.tex (full text).

The layout rules come from the template PDF (ELCVIA_Template.pdf): A4, 11 pt Times, text area 17 x 25 cm with a
2 cm left margin, title 16 pt bold, section titles 14 pt bold, subsection titles 12 pt bold, abstract 10 pt,
a running head on every page after the first, a numbered reference list. The journal's own elcvia.cls is not
available here, so the layout is rebuilt with standard packages.

To stay near the recommended 8-12 pages at 11 pt, the version drops: the accuracy, model and learning-rate-sweep
tables, the 10% per-class table and confusion figure, the per-class heatmap, the reproducibility table (it
becomes a sentence), the discussion subsection that repeats results, and the two long appendices (all runs and
the design decisions, which stay in results/tables/all_runs.md and docs/DESIGN_DECISIONS.md). main_ieee.tex
is the complete version and is never changed here.
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
old = (root / "main_ieee.tex").read_text(encoding="utf-8")


def sub1(pattern: str, repl: str, text: str, flags: int = 0) -> str:
    """Replace exactly one match (the replacement is literal); fail loudly when the pattern is missing."""
    new, n = re.subn(pattern, lambda m: repl, text, count=1, flags=flags)
    assert n == 1, f"pattern not found: {pattern[:70]}"
    return new


def drop(pattern: str, text: str, flags: int = re.S) -> str:
    """Delete exactly one match."""
    return sub1(pattern, "", text, flags)


abstract = re.search(r"\\begin\{abstract\}\n(.*?)\n\\end\{abstract\}", old, flags=re.S).group(1).strip()
keywords = "EuroSAT, Sentinel-2, label efficiency, multispectral, pseudo-labelling, transfer learning"
body = old[old.index("\\section{Introduction}"):]

HEAD = r"""% Report in the layout of the ELCVIA journal template (ELCVIA_Template.pdf): A4, 11 pt Times, 17 x 25 cm text area,
% 2 cm left margin, bold title / section / subsection sizes of 16 / 14 / 12 pt, running head, numbered references.
% The full two-column IEEE version is main_ieee.tex.
% Build: latexmk -pdf main.tex   (pdfLaTeX; needs fancyhdr, titlesec, microtype, booktabs, which Overleaf has)
\documentclass[twoside,a4paper,11pt]{article}
\usepackage[a4paper,twoside,asymmetric,left=2cm,right=1cm,top=2.2cm,bottom=2.5cm,headheight=14pt,headsep=12pt]{geometry}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{array}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage[explicit]{titlesec}
\usepackage[hidelinks]{hyperref}
\newcommand{\TODO}[1]{\textcolor{red}{[TODO: #1]}}
\newcommand{\figdir}{../results/figures}
\newcommand{\tabdir}{../results/tables}

% Running head as in the template: page 1 carries the journal name only; later pages carry
% "author et al. / journal volume(number):pages, year" with the page number on the outer edge
% (left on even pages, right on odd pages).
\pagestyle{fancy}
\fancyhf{}
\fancyhead[CE,CO]{\small\itshape Bhoyar et al. / Electronic Letters on Computer Vision and Image Analysis 0(0):1--12, 2026}
\fancyhead[LE,RO]{\small\thepage}
\renewcommand{\headrulewidth}{0pt}
\fancypagestyle{plain}{\fancyhf{}\fancyhead[C]{\small\itshape Electronic Letters on Computer Vision and Image Analysis}%
  \renewcommand{\headrulewidth}{0pt}}

% Headings: section 14 pt bold, subsection 12 pt bold, subsubsection 11 pt bold (all numbered).
\titleformat{\section}{\fontsize{14}{17}\selectfont\bfseries}{\thesection}{1em}{#1}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{1em}{#1}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection}{1em}{#1}
\titlespacing*{\section}{0pt}{16pt plus 3pt minus 2pt}{7pt plus 1pt}
\titlespacing*{\subsection}{0pt}{11pt plus 2pt minus 2pt}{5pt plus 1pt}
\titlespacing*{\subsubsection}{0pt}{9pt plus 2pt minus 2pt}{4pt plus 1pt}
\setlength{\parindent}{1.5em}
\setlength{\parskip}{0pt plus 1pt}
\raggedbottom
% Let floats sit at the top of a page more freely, so that figures and tables do not leave big gaps.
\setcounter{topnumber}{3}
\renewcommand{\topfraction}{0.9}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.8}

\begin{document}
\thispagestyle{plain}

% Title block: title 16 pt bold, authors with a star, affiliation in italics, a rule, the abstract (10 pt), key words, a rule.
\begin{center}
{\fontsize{16}{19}\selectfont\bfseries Label-Efficient Land-Use Classification on EuroSAT:\\
RGB versus 13-Band Input, Supervised versus Pseudo-Labelled Training\par}
\vspace{14pt}
{\large Ishant Bhoyar$^{*}$, Gaurav Kumar$^{*}$, Jagriti Pandey$^{*}$ and Ayush Kumar$^{*}$\par}
\vspace{9pt}
{\small\itshape $^{*}$ Deep Learning, Sitare University (enrolment nos.\ 202410101200059, 202410101200047,
202410101200060 and 202410101200028)\par}
\vspace{6pt}
{\small Received 22 November 2026\par}
\end{center}
% Journal information at the bottom of page 1, as in the template (no e-mail and no editor name are given).
\begingroup\renewcommand{\thefootnote}{}%
\footnotetext{ELCVIA ISSN:1577-5097\\
Published by Computer Vision Center / Universitat Aut\`onoma de Barcelona, Barcelona, Spain}%
\endgroup
\vspace{8pt}
\noindent\rule{\textwidth}{0.4pt}
\vspace{4pt}

\begin{center}
{\small\bfseries Abstract}
\end{center}
{\small
\begin{list}{}{\setlength{\leftmargin}{1cm}\setlength{\rightmargin}{1cm}\setlength{\topsep}{2pt}\setlength{\parsep}{0pt}}
\item[] ABSTRACT_TEXT
\end{list}
\vspace{6pt}
\noindent\hspace{1cm}\textit{Key Words}: KEYWORDS_TEXT.\par
}
\vspace{8pt}
\noindent\rule{\textwidth}{0.4pt}

"""
head = HEAD.replace("ABSTRACT_TEXT", abstract).replace("KEYWORDS_TEXT", keywords)

# --- one column: starred floats become normal floats ---------------------------------------------------------
body = body.replace("\\begin{figure*}[t]", "\\begin{figure}[t]").replace("\\end{figure*}", "\\end{figure}")
body = body.replace("\\begin{table*}[t]", "\\begin{table}[t]").replace("\\end{table*}", "\\end{table}")

# --- dropped floats ---------------------------------------------------------------------------------------------
body = drop(r"\\begin\{figure\}\[t\]\n\\centering\n\\includegraphics\[width=\\columnwidth\]\{\\figdir/per_class_f1_1pct\.png\}.*?\\end\{figure\}\n", body)
body = drop(r"\\begin\{figure\}\[t\]\n\\centering\n\\includegraphics\[width=0\.85\\textwidth\]\{\\figdir/confusion_10pct\.png\}.*?\\end\{figure\}\n", body)
body = drop(r"\\input\{\\tabdir/per_class_f1_10pct\.tex\}\n", body)
body = sub1(r"\(Fig\.~\\ref\{fig:confusion10\}, Table~\\ref\{tab:per_class_f1_10pct\}\) the picture changes", "the picture changes", body)
body = drop(r"\\input\{\\tabdir/training_cost\.tex\}\n\n", body)
body = sub1(r"Table~\\ref\{tab:training_cost\} gives the mean time per supervised run\.",
            r"""The mean time per supervised run on a free-tier T4 is 4--9 minutes at 1\% and 10\% of the labels and 22--42 minutes at 100\%, depending on the model.""", body)
body = drop(r"\\input\{\\tabdir/supervised_acc\.tex\}\n\n", body)
body = sub1(r"Test accuracy \(Table~\\ref\{tab:supervised_acc\}\) is between", "Test accuracy (table in the repository) is between", body)
body = drop(r"\\begin\{table\}\[t\]\n\\centering\n\\caption\{The three models\..*?\\end\{table\}\n\n", body)
body = sub1(r"We compare three model families \(Table~\\ref\{tab:models\}\)\.", "We compare three model families.", body)
body = drop(r"\\input\{\\tabdir/lr_sweep\.tex\}\n\n", body)
body = sub1(r"\(Table~\\ref\{tab:lr_sweep\}\)\. For the scratch CNN, \$2\\times10\^\{-3\}\$ was best among \$\\\{10\^\{-3\}, 2\\times10\^\{-3\}, 4\\times10\^\{-3\}\\\}\$\.",
            r"""(the sweep is listed in the repository). For the scratch CNN, $2\times10^{-3}$ was best among $\{10^{-3}, 2\times10^{-3}, 4\times10^{-3}\}$, with validation macro-F1 0.930, 0.931 and 0.921.""", body)
body = drop(r"\\begin\{table\}\[t\]\n\\centering\n\\caption\{Steps to reproduce the results\..*?\\end\{table\}\n\n", body)
body = sub1(r"Table~\\ref\{tab:repro\} lists the steps, each a script in the \\texttt\{scripts\} folder of the repository\.",
            r"""The steps are scripts in the \texttt{scripts} folder: \texttt{download\_data.py}, \texttt{make\_splits.py}, \texttt{build\_cache.py}, \texttt{run\_grid.py}, \texttt{evaluate.py} (or \texttt{parse\_colab\_logs.py} and \texttt{eval\_from\_logs.py} when only the run logs exist), \texttt{make\_figures.py} and \texttt{make\_report\_tables.py}. A table of all 51 runs is in \path{results/tables/all_runs.md} and the design decisions with their known weaknesses are in \path{docs/DESIGN_DECISIONS.md}.""", body)

# --- the paragraph that announces the sections (it mentioned the appendices) -------------------------------------
body = drop(r"The rest of the report is organised as follows\..*?\n\n", body)

# --- discussion: drop the answers (they are in Results and Conclusion) and merge the threats into the limitations
body = drop(r"\\subsection\{Answers to the three questions\}.*?(?=\\subsection\{Practical recommendations\})", body)
body = drop(r"\\subsection\{Practical recommendations\}\n", body)
body = drop(r"\\subsection\{Threats to validity\}.*?(?=\\section\{Limitations\})", body)
body = sub1(r"\\item The compute budget was a few free-tier T4 sessions",
            r"""\item We use one dataset with one split, $64\times64$ patches upsampled to $224\times224$, and models pretrained on natural images, so results may differ on larger patches, on other regions, or with models pretrained on satellite data. Macro-F1 is close to saturation at 100\% of the labels, so differences there carry little information.
\item The compute budget was a few free-tier T4 sessions""", body)

# --- appendices: the two long tables go; the reproducibility text becomes a normal section before the references
idx = body.index("\\appendices")
main_part, tail = body[:idx], body[idx:]
repro = tail[tail.index("\\section{Reproducibility}"):tail.index("\\end{document}")].replace("app:repro", "sec:repro")
main_part = sub1(r"\\bibliographystyle\{IEEEtran\}\n\\bibliography\{references\}\n",
                 repro + "\n{\\small\n\\bibliographystyle{plain}\n\\bibliography{references}\n}\n", main_part)
body = main_part + "\n\\end{document}\n"

# --- figure widths for the 17 cm text width -------------------------------------------------------------------------
body = sub1(r"width=0\.85\\textwidth\]\{\\figdir/label_efficiency_f1\.png\}", r"width=0.86\textwidth]{\figdir/label_efficiency_f1.png}", body)
body = sub1(r"width=\\columnwidth\]\{\\figdir/rgb_vs_ms_gain\.png\}", r"width=0.60\textwidth]{\figdir/rgb_vs_ms_gain.png}", body)
body = sub1(r"width=\\columnwidth\]\{\\figdir/pseudo_label_gain\.png\}", r"width=0.72\textwidth]{\figdir/pseudo_label_gain.png}", body)
body = sub1(r"width=0\.85\\textwidth\]\{\\figdir/confusion_1pct\.png\}", r"width=0.90\textwidth]{\figdir/confusion_1pct.png}", body)
body = sub1(r"width=0\.82\\columnwidth\]\{\\figdir/dataset_samples\.png\}", r"width=0.40\textwidth]{\figdir/dataset_samples.png}", body)
body = sub1(r"width=\\columnwidth\]\{\\figdir/dataset_spectral\.png\}", r"width=0.72\textwidth]{\figdir/dataset_spectral.png}", body)

body = body.replace("Fig.~\\ref", "Figure~\\ref")

out = head + body
for leftover in ("tab:training_cost", "tab:supervised_acc", "tab:models", "tab:lr_sweep", "tab:repro", "tab:per_class_f1_10pct", "fig:confusion10",
                 "tab:all_runs", "tab:design", "app:", "appendices", "IEEEtran", "IEEEkeywords", "IEEEauthor"):
    assert leftover not in out.replace("\\label", "").replace("main_ieee", ""), f"leftover {leftover}"
(root / "main.tex").write_text(out, encoding="utf-8")
print("written", len(out.splitlines()), "lines")
