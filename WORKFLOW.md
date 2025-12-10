# Paper Workflow & Git Conventions

Standard workflow for managing academic paper submissions across the highdimensional research projects.

## Repository Structure

Each paper that gets a public GitHub repo follows this structure:

```
paper-name/
├── manuscript_[journal].tex    # Current submission (e.g., manuscript_pre.tex)
├── manuscript_[journal].pdf    # Compiled PDF
├── cover_letter_[journal].tex  # Journal-specific cover letter
├── cover_letter_[journal].pdf
├── references.bib              # Shared bibliography
├── figures/                    # PDF + PNG versions of all figures
├── code/
│   └── simulations.py          # Consolidated simulation code
├── archive/                    # Previous submissions, old code
└── README.md                   # Current status, how to run
```

## Git Workflow

### Public vs Private Repos

- **highdimensional** (private): Umbrella repo for all research projects
- **Public repos**: Created when a paper is ready for submission (for reproducibility)
  - Add folder to highdimensional's `.gitignore`
  - Manage public repo separately

### Naming Conventions

**Manuscripts:**
- `manuscript_[journal].tex` - e.g., `manuscript_pre.tex`, `manuscript_prx.tex`
- When pivoting journals, create new file, archive old one

**Cover letters:**
- `cover_letter_[journal].tex` - e.g., `cover_letter_pre.tex`

### Archiving on Journal Pivot

When a paper is rejected and you're resubmitting elsewhere:

1. Move old manuscript + cover letter to `archive/`
2. Create new `manuscript_[newjournal].tex` with revised content
3. Update README.md with submission history
4. Commit with descriptive message:
   ```
   Pivot to [Journal]: [brief description of changes]

   - Key change 1
   - Key change 2
   - Archived [old journal] files
   ```

## Code Management

### Consolidation

Before submission, consolidate multiple simulation files into single `simulations.py`:
- Easier for reviewers to run
- Fits Gemini's 10-file transfer limit
- Old individual files go to `code/archive/`

### Standard simulations.py structure

```python
#!/usr/bin/env python3
"""
[Paper Name] - Complete Simulation Suite

Usage:
    python simulations.py              # Run all
    python simulations.py [sim_name]   # Run specific simulation
"""

# All simulations as functions
def sim1_name(save_fig=True): ...
def sim2_name(save_fig=True): ...

def run_all():
    """Run all simulations."""
    sim1_name()
    sim2_name()

if __name__ == "__main__":
    # CLI argument handling
```

## LaTeX Build Files

**Always delete before committing:**
- `*.aux`, `*.log`, `*.out`, `*.blg`, `*.bbl`, `*Notes.bib`

**Keep:**
- `.tex` source files
- `.pdf` compiled outputs
- `.bib` bibliography

## Submission History Tracking

In README.md, maintain a submission history table:

```markdown
## Submission History

| Date | Journal | Outcome |
|------|---------|---------|
| Dec 6, 2025 | PRX | Submitted (ID: XM10873) |
| Dec 10, 2025 | PRX | Desk reject |
| Dec 15, 2025 | PRE | Submitted |
```

## Journal-Specific Pivots

When pivoting between journals, common changes include:

### PRX → PRE
- Move derivations from supplementary to main text
- Replace narrative language with rigorous terminology
- Add explicit connections to established theorems (Jarzynski, etc.)
- Descriptive title instead of "branded" title

### Nature family → Specialty journal
- Add more technical depth
- Remove "accessibility" framing
- Cite field-specific literature more heavily

### Specialty → Broader journal
- Add "so what" framing
- Simplify mathematical notation
- Lead with implications, not methods

## AI Workflow Documentation

Each paper should note in README.md:
- Which AI tools were used (Claude Code, Gemini, GPT)
- Role of each (drafting, review, code generation)
- That author takes full responsibility for content

This is disclosed in manuscript acknowledgments/declarations.
