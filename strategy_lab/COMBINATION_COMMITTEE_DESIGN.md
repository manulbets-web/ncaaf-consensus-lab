# Combination Committee / Hypergraph Design

The next ensemble layer should **not** brute-force "combinations of
combinations." That would create a much larger multiple-testing problem and
would let the same base model vote dozens of times simply because it appears
inside many overlapping subsets.

Instead use a **committee of stable combinations**.

## Graph

Treat the system as a bipartite / hypergraph:

```text
base models  ->  candidate combinations  ->  game-side votes
```

A combination node contains 4–8 base-model nodes.

For a given game each combination computes the same fixed statistic:

```text
combo mean margin
combo SD
edge = mean - market
signal = abs(edge) / SD
```

It abstains unless the normal N and k gates pass.

## Prevent combinatorial echo

Two combinations such as

```text
A+B+C+D+E
A+B+C+D+F
```

are not independent voters.

Create a combination-similarity graph using Jaccard overlap:

```text
J(A,B) = |models_A intersect models_B| / |models_A union models_B|
```

Optionally augment this with historical decision/edge correlation.

Cluster highly overlapping combinations into communities. Each **community
gets one vote**, regardless of how many near-duplicate combinations it
contains.

That avoids a model such as Sasser being counted 30 times because Sasser
appears in 30 high-ranked subsets.

## Baseline committee rule

Use the top N combinations selected using training data only.

For each test/current game:

1. score each combination independently;
2. inactive combinations abstain;
3. collapse active combination votes within overlap communities;
4. each active community casts one Home/Away vote;
5. report:
   - active combinations,
   - active independent communities,
   - community vote share,
   - unique base-model support,
   - median combination signal,
   - range of combination edges.

The first version should use **equal community votes**.

Do not weight votes by retrospective ROI. A secondary sensitivity analysis can
weight communities by how often their member combinations recur across
chronological training folds.

## Chronological validation

For every fold:

```text
training weeks
    -> exhaustive combination ranking
    -> freeze top N combinations
    -> freeze overlap graph/communities
    -> next two weeks
    -> committee votes
```

The graph must be rebuilt using training information only.

This tests the actual prospective policy rather than asking whether one exact
combination happened to remain optimal.

## Candidate decision grid

Do not optimize all of these simultaneously at first. A compact grid is enough:

```text
top N combinations:          10, 25, 50
Jaccard community threshold: 0.50, 0.60, 0.70
minimum active communities:  3, 4, 5
community agreement:         0.60, 0.67, 0.75
combo k:                     fixed at the selected primary k
```

The key output is whether committee agreement improves chronological OOS
performance and stability versus the single best combination and the ordinary
base-model mean.


## Implemented in v3.5.8

The first committee implementation uses a rank-ordered 0.60 Jaccard community rule. A combination joins an existing community only when its model membership overlaps that community's best-ranked representative by at least 0.60; otherwise it starts a new community. This avoids transitive graph chaining.

Each frozen finalist is evaluated over k = 0.25, 0.50, ..., 2.00 on discovery data only. The selected k maximizes a neighboring-threshold Wilson-floor criterion (with bet-count gates and volume tie-breaks), rather than maximizing one isolated historical point. The chosen k is frozen before holdout scoring.

The diversified META spread is built in two stages:

1. within each overlap community, average the scorable combination means;
2. average the active community means with equal community weight.

META uncertainty uses total-variance decomposition: average within-unit variance plus between-unit variance. Thus it reflects both disagreement inside combinations and disagreement between independent combination communities. META has its own discovery-selected k and is reported on both discovery and the untouched recent holdout. A naive equal-combination META remains visible as a benchmark.


## v3.5.9 experimental extension: unequal weighting inside finalists

Equal weighting remains the production strategy. For each frozen C1–C12, v3.5.9 compares equal weighting with inverse-MSE reliability weights and a 50%-shrunk reliability variant. Weights are learned on discovery only and frozen for holdout. Each method receives a discovery-selected stable k and is evaluated on holdout using forecast MAE/RMSE plus ATS/ROI. The diversified META layer is rebuilt under each internal weighting method and separately backtested. No holdout result is used to alter the current-week production strategy in this experimental release.
