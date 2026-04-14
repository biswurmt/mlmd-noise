# Guidelines for Final Report Project LMP2392H

**Total pages:** 8 pages (40% of overall mark)

## Title + Team (very briefly)

- Title (ok to modify from proposal as project may evolve)
- Names of students and clinical/senior lead
- Describe contributions of each team member

## Introduction/Clinical Motivation (~¾ page - 5%)

Clearly describe the clinical or health-system problem being addressed:
- Who it affects
- Why it matters in practice
- Ground the problem in a real clinical workflow
- Avoid vague objectives

## Related Work (~¾-1 page - 5%)

Ground work in existing research:
- Introduce existing solutions to the problem (both specific problem solutions and technical solutions)
- Identify solutions that your work builds upon
- Clearly articulate gaps in existing solutions
- Describe innovations you are proposing

## Data & Setting (~0.5-0.75 page)

Describe:
- Data type(s)
- Source (public, synthetic, or hypothetical)
- Sample size
- Key variables and outcomes

## Methods (~1.5 pages)

Describe:
- Primary model(s)
- Rationale for model choice
- Feature representation or embedding strategy
- Training setup
- Hyperparameter selection (high level)

**Note:** Data & Setting and Methods together account for 15% of overall mark.

## Evaluation (~0.5-0.75 page)

Describe:
- Compared baseline approaches
- Primary metrics
  - Standard metrics (DICE, AUROC, AUPR, sensitivity/specificity) do not need definition
  - Justify your metric choices and explain why you selected them
  - Define non-standard metrics
- Fairness analysis where relevant

## Results (~1.5-2 pages)

Include:
- Main quantitative results and their interpretation (!)
- Figures and Tables (especially helpful for baseline comparisons)
- Error analysis (!)
- Statistical uncertainty (confidence intervals, bootstrap, etc.)
- Explainability where possible (Grad-CAM, SHAP, etc. as appropriate)
- Clinical relevance explanation
  - Explain results not just in technical terms but in context of the clinical problem

**Note:** Evaluation and Results sections together account for 10% of overall mark.

## Discussion and Limitations (~0.5-0.75 pages)

- Briefly summarize what you accomplished
- Describe your project's role (decision-support vs. automation)
- Discuss biases and limitations
- Expand on clinical relevance

## Future Work (paragraph)

- Potential impact when executed at scale
- Concrete next steps needed to achieve impact

**Note:** Discussion and Limitations and Future Work together account for 5% of overall mark.

## General Notes

- Section sizes are approximate
- Evaluation and Results sections can be combined
