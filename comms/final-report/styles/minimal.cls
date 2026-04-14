%% minimal.cls
%% Minimalist Academic Template for Milestone Two
%% CSC2541H · AI for Drug Discovery · University of Toronto
%%
%% Based on the milestone_one_minimal design.
%% Provides: 11pt Times New Roman, 2cm margins, navy accents, small-caps headings.

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{minimal}[2026/04/09 Minimalist Template for Milestone Two]

\LoadClass[11pt,a4paper]{article}

%% ── Typography ───────────────────────────────────────────────────────────────
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{newtxtext,newtxmath}
\RequirePackage{microtype}

%% ── Page geometry ────────────────────────────────────────────────────────────
\RequirePackage[top=2cm,bottom=2cm,left=2cm,right=2cm]{geometry}

%% ── Spacing ──────────────────────────────────────────────────────────────────
\RequirePackage{setspace}
\singlespacing
\setlength{\parindent}{1em}
\setlength{\parskip}{2pt}

%% ── Colour palette ───────────────────────────────────────────────────────────
\RequirePackage[dvipsnames]{xcolor}
\definecolor{navy}{HTML}{1A2B3C}

%% ── Section headings ─────────────────────────────────────────────────────────
%% Sections: normalsize bold small-caps navy with rule — prominent.
%% Subsections: small bold small-caps navy, no rule — subordinate.
%% Paragraphs: italic run-in, no vertical space above, indented like body.
\RequirePackage{titlesec}
\RequirePackage{needspace}
\titleformat{\section}
  {\normalfont\normalsize\bfseries\scshape\color{navy}}
  {}{0em}{\Needspace*{4\baselineskip}}
  [\vspace{1.5pt}{\color{navy}\titlerule[0.4pt]}]
\titlespacing*{\section}{0pt}{9pt plus 2pt minus 1pt}{5pt}

\titleformat{\subsection}
  {\normalfont\small\bfseries\scshape\color{navy}}
  {}{0em}{\Needspace*{3\baselineskip}}
  []
\titlespacing*{\subsection}{0pt}{3pt plus 1pt minus 1pt}{2pt}

\titleformat{\paragraph}[runin]
  {\normalfont\small\itshape}
  {}{0em}{}
  [.]
\titlespacing*{\paragraph}{\parindent}{0pt}{0.4em}

%% ── Figures ──────────────────────────────────────────────────────────────────
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=period,skip=3pt}
\newcommand{\diagramwidth}{\textwidth}

%% ── Tables ───────────────────────────────────────────────────────────────────
\RequirePackage{booktabs}
\RequirePackage{tabularx}

%% ── Mathematics ──────────────────────────────────────────────────────────────
\RequirePackage{amsmath}

%% ── Bibliography ─────────────────────────────────────────────────────────────
\RequirePackage[numbers,square]{natbib}
\bibliographystyle{sn-mathphys-num}
\setlength{\bibsep}{3pt}
\providecommand{\bibcommenthead}{}

%% ── Hyperref (load last) ─────────────────────────────────────────────────────
\RequirePackage[
  colorlinks=true,
  linkcolor=navy,
  citecolor=navy,
  urlcolor=navy]{hyperref}

%% ── Title block ──────────────────────────────────────────────────────────────
%% Redefine \abstract{} as a storage command.
%% Define \keywords{} for optional keyword line.
\makeatletter
  \let\@abstracttext\@empty
  \long\def\abstract#1{\long\gdef\@abstracttext{#1}}
  \let\@keywords\@empty
  \def\keywords#1{\gdef\@keywords{#1}}

  \renewcommand{\maketitle}{%
    \noindent{\color{navy}\rule{\textwidth}{1.5pt}}\par
    \vspace{5pt}%
    \noindent{\large\bfseries\@title}\par
    \vspace{4pt}%
    \noindent{\small%
      Tyler Biswurm,\enspace Alice Chua}\par
    \vspace{1pt}%
    \noindent{\small\textit{%
      Department of Computer Science, University of Toronto
      \enspace$\cdot$\enspace CSC2541H
      \enspace$\cdot$\enspace April~2026}}\par
    \vspace{8pt}%
    {\small\noindent\textbf{Abstract.}\enspace\@abstracttext}%
    \ifx\@keywords\@empty\else
      \par\vspace{2pt}%
      {\small\noindent\textbf{Keywords:}\enspace\textit{\@keywords}}%
    \fi
    \par\vspace{6pt}%
    \noindent{\color{navy}\rule{\textwidth}{0.4pt}}\par
    \newpage
  }
\makeatother

\pagestyle{plain}

\endinput
