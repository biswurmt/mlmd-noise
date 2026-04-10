# comms

Communication materials for the Diagnotix project — presentations and pitch content.

## Subfolders

### `class-presentation/`

LaTeX slide deck for the LMP2392 course presentation. Includes:

- `mlmd-presentation.tex` / `presentation.tex` — main Beamer source files
- `outline.md` — slide-by-slide outline
- `desk-reference.md` — speaker notes / Q&A reference card
- `background-compressed.md` — condensed background research summary
- `background/` — full background research documents and the `lilaw/` subfolder with LaTeX assets (figures, style files, bibliography) from the ICML-format proposal template
- `images/` — screenshots used in slides (e.g. `webapp-ss.png`)

Build the PDF:
```bash
cd comms/class-presentation
make          # runs pdflatex + bibtex
```

### `pitch/`

Pitch deck materials:

- `pitch-requirements.md` — pitch brief and evaluation criteria
- `template-slide-formats.md` — slide format templates
- `ideation/` — narrative drafts (`narrative-ed.md`, `narrative-main.md`), pitch outline, and raw monologue transcript
