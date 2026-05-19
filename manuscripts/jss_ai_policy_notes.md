# JSS Submission Strategy & AI Policy Notes

This document outlines the strategy for submitting the `smoothbp` package and manuscript to the **Journal of Statistical Software (JSS)**, specifically regarding the use of generative AI tools.

## 1. JSS & COPE Policy Overview
The Journal of Statistical Software (JSS) follows the **Committee on Publication Ethics (COPE)** guidelines regarding Large Language Models (LLMs) and generative AI:

*   **Authorship:** AI cannot be listed as an author. The human author(s) must be able to take full responsibility for the accuracy and integrity of the work.
*   **Transparency:** Use of AI in drafting text or generating code must be disclosed in the manuscript (typically in the **Acknowledgements** or **Methods** section).
*   **Responsibility:** The author is the final guarantor of the reproducibility of the software and the technical correctness of the manuscript.

## 2. Potential Submission Risks
*   **Code Reproducibility:** JSS technical editors will attempt to compile the Rust backend and run the R examples. Any "hallucinated" dependencies or brittle code will result in immediate rejection.
*   **Literature Accuracy:** AI-generated landscape reviews must be cross-verified to ensure all cited packages exist and are described accurately.
*   **Style and Tone:** The manuscript must adhere strictly to the `jss.cls` LaTeX style and maintain a precise, technical tone.

## 3. Recommended Disclosure & Validation Strategy
To maintain high ethical standards and ensure a smooth review process, the following steps are planned:

### Disclosure Statement
Include the following (or similar) in the manuscript Acknowledgements:
> "The authors acknowledge the use of Project Antigravity (Gemini 3.1 Pro) as an agentic coding assistant for the implementation of the Rust MCMC backend and for drafting initial sections of this manuscript. All code and text were independently reviewed, tested, and validated by the authors. A substantive log of AI interactions is maintained by the authors."

### Validation Checklist
- [ ] **Technical Audit:** Manually run all vignettes (especially `brms-comparison`) to ensure stability.
- [ ] **Mathematical Verification:** Rigorous line-by-line check of the "Statistical Framework" LaTeX equations.
- [ ] **Citation Verification:** Confirm the status and features of `mcp`, `segmented`, and `brms` as described in the "Software Landscape" section.
- [ ] **Source Receipt:** Maintain the `manuscripts/prompt_log_and_acknowledgements.md` file as a local audit record for the editorial board if requested.

### Submission Timing
Check the JSS "Information for Authors" page on the day of submission for any newly updated policies regarding AI, as the regulatory environment for generative tools is evolving rapidly.

---
*Created on: 2026-05-13*
*Tools used for analysis: Antigravity AI (Gemini 3 Flash)*
