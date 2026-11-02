def llm_appendix(tools, text_rows, code_rows, fmt="latex"):
    """Render the 'Use of AI/LLM Tools' appendix as LaTeX (default) or markdown.

    tools:     list of strings, e.g. ["Claude (claude.ai, Sept-Oct 2026)"]
    text_rows: list of (section, level 0-3, note)
    code_rows: list of (file, level 0-4, description)
    """
    if fmt == "latex":
        out = ["\\section*{Appendix A: Use of AI/LLM tools}",
               "\\textbf{A.1 Tools used.} " + ("; ".join(tools) if tools else "None") + ".",
               "", "\\textbf{A.2 Text writing and editing.}",
               "\\begin{tabular}{llp{7cm}}", "Section & Level & Notes\\\\ \\hline"]
        out += [f"{s} & {l} & {n}\\\\" for s, l, n in text_rows]
        out += ["\\end{tabular}", "", "\\textbf{A.3 Code generation and assistance.}",
                "\\begin{tabular}{llp{7cm}}", "File / notebook & Level & Description\\\\ \\hline"]
        out += [f"\\texttt{{{f}}} & {l} & {d}\\\\" for f, l, d in code_rows]
        out += ["\\end{tabular}"]
    else:
        out = ["### Appendix A: Use of AI/LLM tools", "",
               "**A.1 Tools used:** " + ("; ".join(tools) if tools else "None"), "",
               "**A.2 Text**", "", "| Section | Level | Notes |", "|---|---|---|"]
        out += [f"| {s} | {l} | {n} |" for s, l, n in text_rows]
        out += ["", "**A.3 Code**", "", "| File | Level | Description |", "|---|---|---|"]
        out += [f"| `{f}` | {l} | {d} |" for f, l, d in code_rows]
    return "\n".join(out)


# Fill these in for YOUR project -- the entries below are an example only
tools = ["Claude (claude.ai, September 2026)",
         "GitHub Copilot (VS Code, throughout)"]
text_rows = [("Abstract", 1, "grammar check"),
             ("Introduction", 2, "two paragraphs rephrased for clarity; ideas and structure ours"),
             ("Theory / Methods", 0, "--"),
             ("Results", 0, "--"),
             ("Conclusion", 1, "minor language corrections")]
code_rows = [("regression.py", 0, "written independently"),
             ("optimisers.py", 2, "Copilot suggested the Adam bias-correction lines; "
                                 "tested against the closed-form OLS solution"),
             ("resampling.py", 1, "Claude found an off-by-one error in the k-fold split"),
             ("plots.ipynb", 1, "Copilot debugged an axis-indexing bug")]

print(llm_appendix(tools, text_rows, code_rows, fmt="latex"))
print()
print(llm_appendix(tools, text_rows, code_rows, fmt="markdown"))