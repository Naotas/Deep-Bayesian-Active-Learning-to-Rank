# Pair-BALD comparison-partner experiments

This experiment keeps the repository's image-level uncertainty sampling and changes
only how a comparison partner is acquired. Its purpose is to test whether a better
partner choice improves ranking on patients that were not used for training, under
the same cumulative number of relative annotations.

## Relation to Houlsby et al.

The attached paper is Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Máté
Lengyel, *Bayesian Active Learning for Classification and Preference Learning*,
arXiv:1112.5745v1. Page numbers below are the printed PDF page numbers.

- Section 2, p. 3, Eq. (1) defines greedy information-theoretic acquisition as the
  expected reduction in posterior parameter entropy. The same page identifies it
  with the conditional mutual information between the unknown response and model
  parameters.
- Section 2, p. 3, Eq. (2) rewrites the objective in output space as predictive
  entropy minus posterior-expected conditional entropy. The paper calls this
  Bayesian Active Learning by Disagreement (BALD).
- Section 3, pp. 4–5, Eqs. (3)–(5) derive analytic approximations specifically for
  Gaussian-process classification with a probit likelihood and an approximately
  Gaussian posterior. Those formulas are not used in this RankNet experiment.
- Section 3.2, pp. 6–7, Eq. (7) treats a comparison as binary classification on a
  pair. A latent GP utility function predicts `u` over `v` when
  `f(u) + epsilon_u > f(v) + epsilon_v`, with additive Gaussian observation noise.
  The likelihood is probit in the utility difference, with the independent noise
  scale absorbed by rescaling `f`. The induced preference kernel is antisymmetric;
  its covariance and mean are given in the supplementary material on p. 17,
  Eqs. (9) and (10).

The current repository instead uses a DenseNet169 RankNet score, a logistic
comparison likelihood, and MC Dropout as an approximate posterior. It has no
separate additive Gaussian observation-noise parameter corresponding to Eq. (7);
uncertainty is represented by the logistic comparison and approximate weight
posterior. For aligned Dropout sample `t`, acquisition uses

```text
p_ij^(t) = sigmoid(f_t(x_i) - f_t(x_j))
pbar_ij = mean_t p_ij^(t)
h(p) = -p log(p) - (1-p) log(1-p)
```

Predictive entropy is `h(pbar_ij)`. Pair-BALD is
`h(pbar_ij) - mean_t h(p_ij^(t))`, which is the Monte Carlo application of Eq. (2).
The implementation deliberately averages probabilities after the sigmoid; it does
not use the sigmoid of an average score difference.

## Existing pipeline behavior confirmed from code

The following behavior comes from scripts 1–6 and the preprocessing code rather
than from the README alone.

- An image ID is the unique JPEG `filename`. Duplicate output filenames are removed
  during preprocessing.
- Fold splits are made by patient. For fold `k`, patient group `k` is test, the next
  group is validation, and the remaining three groups are training. The split seed
  is `20191125`.
- Pair CSV columns are `x1_image`, `x2_image`, `x1_label`, `x2_label`, and
  `relative_label`. The target is 1 when the first Mayo grade is higher, 0 when it
  is lower, and 0.5 when grades are equal.
- AL_0 selects 20% of train and validation images and creates one derangement pair
  per selected image. Later rounds select 5% of the full split by prediction-score
  variance (UBS), excluding images previously selected as anchors.
- The released script 5 recreates one random derangement over the entire cumulative
  selected-image set. It does not preserve the earlier round's pair rows.
- Checkpoints are Keras HDF5 weight files named
  `epoch<epoch>-<val_loss>.h5`. `save_best_only=True` means the lexicographically
  last saved epoch is the final best checkpoint in the existing scripts.
- Script 6 does not continue from the preceding AL model. Every AL round loads the
  same fixed AL_0 checkpoint, then trains independently on that round's cumulative
  pair CSV. This initialization policy is retained.
- Dropout layers are constructed with `training=True`, including Dropout inside the
  DenseNet. BatchNormalization is called normally and therefore uses inference
  statistics during `model.predict`.
- Existing MC prediction CSVs store `filename`, `label`, and `sampling_0` through
  `sampling_29`. Because ordinary Dropout samples a mask across the batch dimension,
  matching those columns across separately processed images does not establish a
  common model-parameter sample for Pair-BALD.

## Experiment definition

The four independent methods are:

1. `original`: uniform partner selection within the cumulative images selected by
   image-level active learning. This preserves the original restricted comparison
   range while acquiring only new pairs so its budget is comparable.
2. `random`: uniform partner selection from the full fold-specific training split.
3. `predictive_entropy`: the full-training candidate with maximum `h(pbar)`.
4. `pair_bald`: the full-training candidate with maximum Pair-BALD.

The original released script also re-pairs every cumulative selected image in every
round. Doing that would spend a different number of new annotations, so this
baseline retains its cumulative-selected candidate pool but applies the common exact
new-pair budget to the current round's anchors. `original` and `random` isolate the
effect of candidate-range expansion. `random`,
`predictive_entropy`, and `pair_bald` share the full-training range and isolate the
acquisition score. Any non-original method can also use the current-round
`selected` scope for a separate diagnostic.

Candidates never include validation or test images. Previously seen images remain
eligible as partners. Self-comparisons, every previously acquired unordered pair,
and duplicates within the current round are excluded. `(i, j)` and `(j, i)` have
the same acquisition key.

The default new-pair budget equals the number of current anchors. A first pass gives
every anchor one feasible partner. Any remaining budget is assigned to the best
currently feasible pair across anchors. A nonzero partner cap limits cumulative
appearances in the `x2_image` role; zero, the main-experiment default, leaves the
cap open so partner concentration can be measured. The command stops with an
explanation if anchor coverage or the exact budget cannot be satisfied.

Mayo labels are not passed to partner ranking. After all pairs are fixed, a separate
oracle step reads the training CSV and creates RankNet targets. Filename text and
directory names are never used as features or candidate filters.

## Aligned MC Dropout approximation

Acquisition inference packs the MC dimension into each model call. Every Dropout
layer generates a stateless mask of shape `T × 1 × ...`: the `T` masks differ, but
the image dimension is shared. Repeating the same stateless generation in later
chunks reproduces the same mask, so `f_t(x_i)` and every `f_t(x_j)` use the same
Dropout realization across chunk boundaries. BatchNormalization remains in
inference mode, and its moving statistics are checked before and after acquisition.

Only the aligned `T × N` image-score cache is stored. Pair probabilities are
calculated for one anchor and one candidate chunk at a time; no `N × N × T` tensor
is constructed. Cache metadata records the ordered image-ID digest, fold,
checkpoint path/size/time/SHA-256, `T`, seed, image shape, Dropout mode, and
BatchNormalization mode. A mismatch stops reuse.

The approximation still has limits. MC Dropout is only an approximate posterior,
and shared activation masks are used as a function-consistent model draw even
though training used ordinary per-example activation masks. Results therefore do
not inherit the exact GP posterior interpretation from Houlsby et al. Finite `T`
also introduces Monte Carlo error. Roundoff-sized negative BALD values up to the
configured tolerance are clipped to zero; larger negative values stop the run.

## Equal-grade comparisons

The training target 0.5 is unchanged. In the binary acquisition model, however,
`p = 0.5` means uncertainty about comparison direction; it is not the probability
of a third, equal-grade answer. Pair-BALD is therefore a binary proxy acquisition
function for a RankNet dataset that includes ties. A ternary observation model is
outside this experiment.

For evaluation, equal-grade pairs are excluded from directional accuracy exactly as
in the existing evaluation. They are reported separately by the mean absolute
distance of `sigmoid(mean_score_i - mean_score_j)` from 0.5. Overall, adjacent,
M0–M1, M1–M2, and M2–M3 directional accuracy are saved against cumulative pair
count.

## Outputs

Every selection directory contains new and cumulative pair CSVs, new and cumulative
anchor lists, the candidate scope, per-candidate `pbar`, predictive entropy, mean
conditional entropy and BALD, experiment conditions and timings, pair counts,
unique-image and per-image appearance counts, Mayo distribution, pair-distance
fractions, and connected-component count. Diagnostic runs omit the cumulative
training artifact. Training, prediction, and evaluation outputs are stored below
the same experiment ID without touching legacy `Add_dataset` or `Results` paths.
