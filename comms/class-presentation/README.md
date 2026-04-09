# class-presentation

LaTeX/Beamer slide deck for the LMP2392 course presentation on Diagnotix.

## Files

| File | Description |
|------|-------------|
| `presentation.tex` | Main Beamer slide deck source |
| `mlmd-presentation.tex` | Alternate/draft version |
| `outline.md` | Slide-by-slide outline |
| `desk-reference.md` | Speaker notes and anticipated Q&A reference card |
| `background-compressed.md` | Condensed background research (for slide content) |
| `background/` | Full background documents: `final-proposal.md`, `lilaw-briefing.md`, `relevant-work-compressed.md`, and `lilaw/` with ICML-format LaTeX assets |
| `images/` | Screenshots used in slides (e.g. `webapp-ss.png`) |

## Build

```bash
cd comms/class-presentation
make
```

The `Makefile` runs `pdflatex` + `bibtex` to produce the PDF. Requires a LaTeX distribution
(e.g. MacTeX, TeX Live).
