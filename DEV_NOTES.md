# Developer Notes: Dynare Macroprocessor & DynoModel Integration

## 1. Dynare Preprocessor vs. DynoModel Preprocessor

Dyno includes a self-contained, pure-Python implementation of the Dynare macro processor in `src/dyno/dynare/macro.py` based on Lark.

While its language grammar, operators, evaluation semantics, built-ins, and loop/conditional directives match the Dynare preprocessor (`dynare-preprocessor <file> onlymacro savemacro=<outfile>`), there are key operational differences in workflow and downstream consumption:

### A. Whitespace & Preservation Behavior
- **Dynare C++ Preprocessor (`onlymacro savemacro=...`):**
  - Always runs a postprocessing step on output:
    1. Strips `@#line ...` directives.
    2. Strips leading empty lines.
    3. Collapses multiple consecutive newlines (`\n{2,} -> \n`).
  - *Consequence:* Even if a `.mod` file has **zero** macro directives, running Dynare's macro preprocessor changes the file's whitespace and blank lines.
- **Dyno Preprocessor (`expand_macro` / `DynoModel(..., preprocess=True)`):**
  - In `only_if_needed=True` mode (used by default by `DynoModel`), Dyno scans the file for macro constructs (`@#`, `@{`, or line continuation `\\`).
  - If **no macro directives are present**, the file content is returned **strictly unchanged**, avoiding any loss of formatting or whitespace.
  - When directives are present, Dyno applies the exact same postprocessing regex pipeline as Dynare to guarantee 1-to-1 byte matching for files using macro directives.

### B. Downstream Parsing & Model Construction
- **Dynare:**
  - Compiles the preprocessed text through Bison/Flex into C++ AST structures, writing code files or emitting JSON payloads (`transformed_modfile`).
- **Dyno (`DynoModel`):**
  - Passes preprocessed `.mod` text into Dyno's Lark-based `LModFile` parser.
  - Generates symbolic equations, context (variables, parameters, shocks), and analytical derivatives natively in Python without requiring compiled C++ libraries.

---

## 2. The Line Number Loss Problem & Proposed Solutions

### The Problem
When macro expansion is performed:
1. **Loop expansion (`@#for ... @#endfor`):** Repeats blocks, injecting many lines into the expanded output that did not exist in the source file.
2. **Conditional blocks (`@#if ... @#endif`):** Prunes false branches, eliminating lines from the source.
3. **File inclusions (`@#include "..."`):** Inlines external files, shifting all downstream line numbers.
4. **Blank line collapsing:** Collapsing `\n{2,} -> \n` shifts line numbers.

As a result:
- AST nodes in Lark record line numbers corresponding to the **expanded** `.mod` stream, not the **original** source file.
- Error messages, residual diagnostics (`results.add_warning(..., line=tree.meta.line)`), or equation tag viewers refer to line numbers in the temporary expanded stream rather than the user's authored file.

---

### Suggested Architectural Approaches

Here are three practical approaches to maintain or restore source line attribution:

### Approach 1: `@#line` Directive Emission & Lark Pre-Lexer Tracking (Recommended)
This is how standard C preprocessors (cpp) and Dynare's native preprocessor handle source tracking.

- **How it works:**
  1. During expansion, `MacroProcessor` emits line directives into the expanded stream:
     ```dynare
     @#line 42 "original_model.mod"
     ```
     whenever line continuity changes (after `@#for`, `@#include`, `@#if`, or skipped blocks).
  2. Dyno's `LModFile` tokenizer or Lark pre-lexer intercepts lines matching `@#line <num> "<file>"`:
     - It updates internal lexer state: `current_line = <num>` and `current_filename = "<file>"`.
     - It drops the line from the token stream so grammar rules don't need modification.
  3. Every Lark AST node's `meta.line` automatically reflects the original source line number and file path.

### Approach 2: Source Map Table (Index Mapping)
- **How it works:**
  1. `MacroProcessor` produces both `expanded_text` and a line-mapping array:
     ```python
     # expanded_line_number -> (source_file, source_line_number)
     sourcemap: list[tuple[str, int]]
     ```
  2. Whenever an error or warning is emitted at line $L$ of the expanded text, Dyno looks up `sourcemap[L]` to report:
     `"Error in original_model.mod at line 35"`.
- **Pros:** Does not require altering Lark grammar or lexer token streams.
- **Cons:** Only tracks line-level resolution (not column-level) within multi-line statements.

### Approach 3: Two-Tier Diagnostics
- **How it works:**
  1. Keep both the original raw text and the expanded text on the model (`model._source_raw` and `model._source_expanded`).
  2. In user-facing reports and GUI explorers, display:
     - The expanded equation with its expanded line index.
     - A snippet of the actual equation text rather than just a line number.
- **Pros:** Completely robust even if line mapping has edge cases in complex nested loops.
