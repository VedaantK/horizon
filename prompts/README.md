# Prompting AI tools to write Concordance strategies

Any model (Claude Code, Codex, or anything else) can write a runnable strategy. The format
doesn't depend on the model being careful; three deterministic pieces enforce it:

1. **A structured spec.** The idea becomes a `StrategySpec` JSON object
   (`strategy_spec.schema.json`). The tools are forced to emit exactly that shape:
   `claude -p --json-schema`, `codex exec --output-schema`.
2. **One contract.** `docs/STRATEGY_CONTRACT.md` is the only description of the file format,
   the data and the order types. Every authoring prompt points at it. Change the contract,
   and every model's instructions change with it.
3. **A validator in the loop.** `concordance validate strategy.py` checks the file
   statically and by running it (synthetic markets, determinism, look-ahead, all three
   engines). The authoring tool runs it itself and fixes what it reports. Then the driver
   runs it again and doesn't trust the tool's claim. Errors carry line numbers and hints, so
   repair rounds converge.

```
idea.txt ──formalize──▶ spec.json ──(you answer open_questions)──▶ author ──▶ strategy.py
                         schema-checked                                   validated
                                                                             │
                     review.md ◀── metrics ◀── concordance backtest ◀───────┘
```

## Commands

```bash
# 1. Idea -> spec. Read open_questions / inferred_defaults / unsupported before going on.
concordance formalize idea.txt -o work/spec.json --tool claude

# 2. Spec -> validated strategy, once per tool, each in its own workspace.
concordance author work/spec.json --tool claude --out candidates/idea_claude
concordance author work/spec.json --tool codex  --out candidates/idea_codex

# 3. Backtest each candidate on the research period (never the lockbox).
concordance backtest request.json -o runs/idea_claude_r1
```

`author` refuses a spec with unanswered `open_questions` unless `--allow-open-questions` is
given. That's deliberate: ambiguity is where implementations diverge.

## Files

| file | used by | output |
|---|---|---|
| `formalize.md` + `strategy_spec.schema.json` | `concordance formalize` | StrategySpec JSON |
| `author.md` + `author_result.schema.json` | `concordance author`, round 1 | `strategy.py` + a summary |
| `repair.md` | `concordance author`, later rounds | fixed `strategy.py` |
| `review.md` + `review.schema.json` | the coordinator (adversarial review) | verdict + up to 3 proposals |

The schemas use the strict subset that OpenAI structured outputs accept: every property is
required, optional ones are nullable, and there are no free-form objects. So the same file
works for Claude and Codex.

## How each tool is run (see `src/concordance/authoring/drive.py`)

| | Claude Code | Codex CLI |
|---|---|---|
| headless | `claude -p` (prompt on stdin) | `codex exec -` (prompt on stdin) |
| structured answer | `--output-format json --json-schema <schema>` → `structured_output` | `--output-schema <file> -o <file>` |
| file access | `--permission-mode acceptEdits` in the workspace | `--sandbox workspace-write --cd <workspace>` |
| commands | only `Bash(<venv>/bin/concordance validate strategy.py:*)` | sandboxed, inside the workspace |
| no state leaks | `--no-session-persistence` | `--ephemeral` |
| billing | `ANTHROPIC_API_KEY` removed from the env → subscription login | `OPENAI_API_KEY` removed → ChatGPT login |

The workspace holds only `spec.json`, `STRATEGY_CONTRACT.md` and `example_strategy.py`. The
tools never see the data lake, the ledger, other candidates or held-out results.

## Do you need a local LLM?

Not for format. Format is guaranteed by the schema plus the validator, not by the model.
A local model (e.g. a 7-14B coder through Ollama) would only help as a fallback author
when both subscriptions hit their usage limits. It would need more repair rounds and
would make more silent logic mistakes (raw vs total-return prices, NaN handling), which
the validator can't catch. If you add one, add it as a third `--tool` in `drive.py`; the
prompts and validator stay the same.
