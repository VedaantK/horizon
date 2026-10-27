Implement the StrategySpec in `spec.json` as a Horizon strategy file named
`strategy.py` in the current directory.

1. Read `STRATEGY_CONTRACT.md` (the contract: file shape, the `view` data, the `Orders`
   API and how orders fill) and `example_strategy.py` (a complete working example).
2. Write `strategy.py`. Follow the spec exactly. Where the spec is silent, choose the most
   conservative reading and list it in `assumptions`. Anything you could not implement as
   written goes in `deviations_from_spec`.
3. Run `{validate_cmd}` and fix every error it reports. Repeat until it prints
   `"ok": true`. Warnings are allowed, but address speed warnings.
4. Do not edit any file other than `strategy.py`. Do not try to improve the strategy;
   implement the spec.

Finish with the JSON object required by the output schema (`params_json` is `self.params`
encoded as a JSON string).
