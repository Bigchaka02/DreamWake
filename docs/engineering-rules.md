# DreamWake engineering rules

## Source
- superpowers https://github.com/obra/superpowers @ 22f0b46 (skills/*/SKILL.md, paraphrased)
- ponytail https://github.com/DietrichGebert/ponytail @ 96608e0 (skills/ponytail, ponytail-review, ponytail-audit, paraphrased)

## Before writing code
1. Restate the task in your own words in 2-4 lines: what, for whom, why. If the why is missing, ask the manager.
2. Read the code the change touches and its callers and tests before designing anything.
3. Ask a question only if the answer changes what gets built and the repo cannot answer it; otherwise state your assumption in one line.
4. For anything beyond a trivial edit, list 2-3 approaches with one-line trade-offs, pick one, and say why.
5. Write down goals and non-goals; anything not a goal is not built (YAGNI).
6. List every file you will create or change, and every caller, test, fixture, or doc the change must reach.
7. Split work into tasks that each end in one independently testable result; a task is one test cycle, not one file.
8. Pin exact function names, signatures, and values in the task text so another agent could implement it without guessing.
9. Do no implementation until the manager has accepted the approach; reading and exploring are always allowed.
10. A throwaway spike is allowed only if labeled "spike"; keeping spike code means going back through steps 1-9.

## Writing code
1. No production code without a failing pytest first. Code written before its test is deleted and redone from the test.
2. Red: write one test for one behavior with a name that states the behavior. Run it and read the failure; it must fail because the feature is missing, not from a typo or import error.
3. Green: write the least code that passes that test; do not add options, parameters, or branches no test demands.
4. Refactor only while green and only to remove duplication or fix names; no new behavior.
5. Tests use real code and real tiny inputs; mock only clocks, randomness, or network. Never assert on a mock's own behavior.
6. Bug fix = failing regression test first, then the fix. Prove it: revert the fix, the test must fail, restore, it passes.
7. Decision ladder, take the first rung that fully works:
   - a. Does it need to exist? If nobody asked, skip it and name it in one line.
   - b. Does this repo already have a helper or pattern? Reuse it the way neighbours do.
   - c. Does the Python 3.14 stdlib do it? Use it. Runtime code is stdlib-only.
   - d. A new runtime dependency is never added without the manager's written yes. pytest is the only dev dependency.
   - e. Otherwise write the minimum code, one clear line over a clever one.
8. No wrapper, base class, factory, config key, flag, or "for later" hook with a single user or no user.
9. Deletion beats addition; moved or merged code keeps its validation and error handling.
10. Keep validation at trust boundaries (files, CLI args, external data), error handling that prevents data loss, and anything the task asked for.
11. A deliberate shortcut gets a one-line comment: `shortcut: <the limit>, <when to upgrade>`.
12. Comment only the why that the code cannot show. Follow existing file structure and naming; do not restructure outside the task.
13. Fix a bug in the shared function after grepping all its callers, not in one caller.

## Debugging
1. No fix before a root cause. Read the whole traceback and error message first.
2. Reproduce it with exact steps or a failing test; if you cannot reproduce, gather data instead of guessing.
3. Check what changed recently (diff, new input, new environment) and find a working similar case to compare against.
4. Trace the bad value backwards through the call chain to where it first went wrong; fix there, not where it surfaced.
5. State one hypothesis ("X is the cause because Y"), test it with the smallest change, one variable at a time.
6. If the hypothesis fails, form a new one; never stack fixes on top of a failed one.
7. Write the failing test before the fix; then one fix, no "while I'm here" edits, then run the full suite.
8. After 3 failed fix attempts, stop and report to the manager that the design may be wrong; do not try a 4th.
9. If you do not understand something, say so in the report. Never present a guess as a finding.

## Before saying it is done
1. Run the full suite (bare `python -m pytest`) fresh, in this session, after your last edit; a single-file run is not enough.
2. Quote the result line (passed/failed counts, exit code) and name every failing or warning test, including ones you did not cause.
3. "Should pass", "looks right", and an earlier run are not evidence. No success wording without a command run after the last change.
4. For a bug fix, show the red-then-green evidence (failing output before, passing after).
5. Re-read the task requirements and tick each one against code or test output; list any gaps or skipped items.
6. Run `git status` and `git diff --stat`; confirm only intended files changed and nothing outside the task.
7. End the report with one or two lines: what you skipped or did not check, and any risk the manager must know.
8. A manager never trusts "success" from a subagent: it checks the diff and reruns the suite itself.

## Review checklist (reviewer subagent answers yes/no to each, with file:line for every "no")
1. Does the diff do exactly what the task asked, with nothing extra and nothing missing?
2. Does each new behavior have a test that would fail if the behavior broke (not a mock assertion)?
3. Do the empty, zero, one-item, last-item, and bad-input cases give a sensible result instead of a crash or silent wrong answer?
4. Is every caller of each changed function or signature still correct (grepped, not assumed)?
5. Are errors handled so data is not lost or silently swallowed, and is external input validated at the boundary?
6. Is the runtime still stdlib-only, and does the code use stdlib/existing helpers rather than a rewrite or new dependency?
7. Is there any abstraction, option, wrapper, or dead code with at most one use that can be deleted without losing behavior?
8. Are near-duplicate blocks merged, and does each function do one job (split by job, not by length)?
9. Does it follow the existing structure and naming, with no unrelated refactors in the diff?
10. Is any deliberate shortcut marked `shortcut:` with its limit, and is the full suite output shown clean?

Reviewer output: Critical (bug, data loss, broken), Important (missing test, wrong edge case, needless code), Minor (polish). Each finding needs a concrete failing input or situation; no input, no finding. Say what you did not read.

## How a manager should brief a subagent
1. One task per subagent; give it the goal, the exact files, the exact signatures/values, and these rules by path (docs/engineering-rules.md).
2. Give only what the task needs, not the whole conversation history; hand long material over as a file path.
3. Set the model explicitly, and the constraints: no new dependencies, no edits outside listed files, no subagents of its own, no commits unless told.
4. Never run two implementers at once on the same files; parallel agents need disjoint files and no shared state.
5. Tell it to ask or report BLOCKED / NEEDS_CONTEXT rather than guess, and to return status, files changed, test summary line, and concerns in under 15 lines.
6. After it reports, verify the diff and rerun the suite yourself, then send a separate fresh reviewer; fix Critical and Important findings before the next task.
