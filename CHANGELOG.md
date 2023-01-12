# Changelog

## v0.3.0

- `compare_forecasters` now returns a `pd.DataFrame`.
- `confseq_**` methods now return a new `ConfSeq` class 
  (a `NamedTuple` with attributes `lcbs` and `ucbs`).
- Various plotting updates, including:
    - Default plotting style is updated (font, size, x/y ranges, etc.).
    - The default synthetic data experiment now includes `mix_01` and `mix_10` forecasters.
    - `plot_comparison` now includes e-processes by default.
    - The x-axis (time) is now in natural scale by default (switched from log).
    - The name of the outcome column is now `y` (switched from `data`).
- Winkler score is also implemented as a class object: `WinklerScore`.
- Winkler score is now one-sided by default (see updated paper).
- E-processes for lagged forecasts are newly implemented (see updated paper).
- Reflects other changes for the [`arxiv.v5` revision](https://arxiv.org/abs/2110.00115).

## v0.2.0

- new `ScoringRule` API w/ support for categorical outcomes
- updated plots
- automatic choice of bounds for CSs
