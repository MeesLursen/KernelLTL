# Experiment 2: estimands, adjustment sets, and the assumptions each output rests on

Companion to `run_exp2.py` and the causal diagram (`misc/Causal_Diagram_EXP2.png`).
One entry per model and exhibit: what it estimates, why its adjustment set is
what it is, which assumptions the reading rests on, and what may and may not
be claimed from it. Written as the working guide for the thesis chapter that
motivates and defends the modelling choices.

Node abbreviations (DAG name = code name): S = operator-presence vector
(`has_op` x 8), D = depth (cell means), V = variance (`z_variance`),
u = studentised variance-binned log-norm residual, F = relational
faithfulness (`z_faith` in models; raw Spearman in curves), C = correctness
(greedy semantic equivalence).

---

## 1. Units and the status of "causal" language

- **The unit of observation is the target formula phi.** Every covariate is a
  deterministic functional of phi given the frozen pipeline (anchor set, trace
  sample, trained decoder): V = p(1-p) of phi's satisfaction vector, u and F
  are functionals of the embedding emb(phi) = F_c satvec_c / N, S and D are
  syntax functionals. The only exogenous randomness is the generator's draw
  of phi. phi itself is never a model variable: with one row per formula it is
  the sample point, and "adjusting for phi" is vacuous.
- **All effect language is population-contrast language.** "The effect of F"
  means: the difference in expected correctness between generator-produced
  formula populations that differ in F but are matched on the adjustment set.
  No claim of the form "changing this formula's F" is made anywhere; syntax
  attributes in particular admit no well-defined intervention (a formula's
  depth cannot be moved while holding its operators fixed in general).
- **Outcome mechanism (two channels).** C = 1[gen(emb(phi)) == phi]. phi
  enters twice: through the embedding the decoder conditions on (the
  *information channel*; the shuffle null certifies correctness collapses to
  chance without it) and through the semantic-equivalence check against phi
  itself (the *difficulty channel*: a deep target is harder to hit even from
  a perfect embedding). The geometry covariates {V, u, F} summarise the
  information channel; {S, D} proxy the difficulty channel. This is the
  mechanistic reading of the DAG's direct S -> C and D -> C arrows.

## 2. The DAG's geometry edges (two dashed, one directed)

The two dashed edges are bidirected edges in the standard latent-projection
sense: an unobserved common cause projected out of the graph, never a claim
that either endpoint causes the other, and never something to "model".

- **S <-> D**: the generator's formula draw induces joint dependence between
  operator presence and depth (more internal nodes make both more likely).
  Harmless for every estimand below because S and D are always conditioned
  together from M2 onward; the edge costs collinearity (precision), never
  validity.
- **F <-> u**: both are coarsenings of the *same* embedding vector -- u is a
  pure **scale** statistic (within a variance bin, a monotone function of
  ||emb(phi)|| alone; `frame.derive_covariates`) and F is a pure **direction**
  statistic (Spearman is invariant to positive rescaling of emb(phi);
  `compute_features._spearman_rows`). Neither enters the other's construction,
  so a directed edge in either direction is excluded *by construction*; their
  dependence runs through the joint law of (scale, direction) -- the latent
  "registration strength relative to the trace-sampling noise floor" (call it
  U_geo). This is a constructional fact (C2 below), not an assumption.

One edge one might expect dashed but which is DIRECTED: **V -> F**. Its
magnitude ("scale") component is provably zero -- std(phi) multiplies the
whole covariance row k_true[phi,.] by a single landmark-independent scalar
(Cauchy-Schwarz), the embedding side scales the same way, and Spearman is
rank-invariant to a global positive scalar; so the obvious channel washes
out exactly as it does for u. But a second, variance-specific channel
survives and is genuinely directed: variance sets the signal-to-noise ratio
of the finite-N covariance estimates (signal ~ std(phi), noise floor roughly
formula-independent), and a Spearman correlation of two noisy-rank vectors is
attenuated toward zero IN EXPECTATION. So low variance lowers E[F], a real
shift in the F value driven by the variance value -- the "reliability
channel" (C4). This channel is localised on variance and is absent for u by
construction (u is within-variance-bin, so SNR is constant along it): the
same construction that makes u variance-orthogonal (C1) puts the entire
reliability channel on the variance node. That asymmetry is why V -> F is
solid while F <-> u is dashed. The originally-floated justification for the
edge (deeper/harder semantics are less faithful) is the shared-latent story
and belongs to the bidirected part; the reliability channel is the directed
part we actually defend, with the artifact caveat in C4.

The remaining geometry edges all point the *same* way -- S, D -> {V, F, u} --
and that orientation is not a modelling choice but the computation order of
the pipeline (C5): the syntax tree is what the grammar samples, the
satisfaction vector is computed *from it* by LTL semantics over the fixed
trace set, and the embedding (hence u and F) is computed *from* the
satisfaction vector. There is no coherent backward operation ("hold the
syntax, change the variance": variance is a *function of* the syntax), so
these edges can be neither reversed nor rendered bidirected-only in the
V <-> F sense -- unlike the two dashed edges, they carry a genuine directed
mechanism. This orientation is load-bearing: every rung conditions on S, D to
*close backdoors* for the geometry covariates, which is valid only because
S, D are upstream confounders and not downstream mediators or colliders.

Because every pair of nodes shares the ancestor phi, **every absent edge is a
claim**. The absences that carry identification weight are A1-A3 below; the
absence of V -> u is the one absence that needs no assumption at all (C1),
and the *orientation* of the S, D edges needs none either (C5).

## 3. Assumption register

Constructional facts (hold by construction; cite, do not defend):

- **C1 (u is variance-orthogonal).** u = (log||emb|| - binned mean) / binned
  SD within 50 variance-quantile bins removes the mechanical Cauchy-Schwarz
  dependence of the norm on variance in location and scale. The missing
  V -> u edge is enforced, not assumed.
- **C2 (scale/direction split).** F is invariant to positive rescaling of
  emb(phi); u depends on emb(phi) only through its norm. Hence no directed
  F--u edge either way; the association is pure shared-latent (U_geo).
- **C3 (determinism).** All covariates are exact functionals of the realised
  pipeline. Under this reading u and F are measured *without error* as
  properties of what the decoder actually consumed; "measurement error" talk
  applies only if effects are interpreted against underlying semantics rather
  than the realised trace sample (see S4).
- **C4 (variance -> faithfulness: scale zero, reliability directed).** The
  scale component of V -> F is zero by the C2 rank-invariance argument. The
  surviving directed component is the reliability channel: variance sets the
  covariance-estimation SNR, and rank-noise attenuates Spearman F toward zero
  in expectation, so E[F] falls with variance. This makes F a genuine (if
  partly artifactual) MEDIATOR of V, which is what licenses excluding F from
  M2 to read V's total effect (Section 4, Q1). *Artifact caveat*: part of
  "low-variance targets are less faithful" is "faithfulness is harder to
  MEASURE at low variance" -- a statement about the estimator, not the
  representation. Its weight as a channel of V's effect on C is only as large
  as F's own link to C; if beta_F is null the caveat is moot. Unlike C1-C3
  this is a mechanism claim, not a pure identity: it is defended, not merely
  cited. The competing reading (V <-> F bidirected, no directed component) is
  adjudicated empirically, not a priori -- see the Q1 note in Section 4.
- **C5 (syntax is computationally prior).** S and D are pure functions of the
  formula *string* (`frame.parse_depth`, `has_op` token presence), fixed the
  instant phi is drawn and computed with no traces at all; variance, u, and F
  are all computed *downstream* of the syntax via the deterministic pipeline
  syntax -> satvec -> {variance; embedding -> u, F}. Hence the S, D -> {V,F,u}
  edges are DIRECTED and their orientation is exact (the computation order),
  and S, D are ROOTS of the modelled system: nothing in {V, F, u, C} feeds
  back into them (the formula is fixed before scoring). Their mutual
  dependence and shared generator origin is the S <-> D dashed edge. This is
  an *orientation/priority* fact and is CERTAIN; it is strictly distinct from
  A3, which concerns *sufficiency*. The coarsening caveat qualifies A3, not
  C5: S and D are coarsenings of the tree, so the edges carry only PART of
  phi's upstream influence (the rest is U_syn / U_geo) -- but a coarse marker
  of an upstream cause is still upstream, so incompleteness of the block never
  threatens the direction of its arrows.

DAG-level assumptions (the load-bearing absences):

- **A1 (geometry sufficiency).** U_geo has no arrow to C: unmodelled
  embedding structure affects correctness only through F and u (given S, D,
  V). *Strictly false* -- three summaries cannot exhaust an embedding -- so it
  is a sufficiency frontier, partially discharged by (i) the shuffle null
  (no information channel outside the embedding) and (ii) S and D proxying
  residual embedding structure. Every causal reading of beta_u and beta_F
  (M3), and the direct-effect reading of beta_V (M3), is conditional on A1.
- **A2 (no variance-difficulty confounding).** No latent trait of phi drives
  both V and C beyond S and D (candidate violation: long-horizon G-heavy
  semantics with extreme p that are also intrinsically hard to reproduce).
  The total-effect reading of beta_V (M2) is conditional on A2 -- and on
  nothing else: with F unconditioned, the collider at F blocks every U_geo
  path, so this reading survives even a failure of A1.
- **A3 (syntax sufficiency).** S and D close the syntax backdoors: no
  residual syntax trait (formula size at fixed depth, nesting shape, atom
  multiplicity) confounds the geometry-correctness relations. Presence
  indicators are coarse (they ignore counts), so A3 is an approximation;
  it is shared by every rung from M2 up. *Distinct from C5*: C5 fixes that
  S, D sit UPSTREAM of the geometry (certain, by construction); A3 is the
  separate, approximate claim that they capture ENOUGH of the upstream syntax
  to close the backdoors. Do not let the certainty of the former lend
  borrowed confidence to the latter.

Statistical assumptions and conventions:

- **S1 (functional form).** Continuous covariates enter linearly on the logit
  scale. Checks: the u-decile curve and the M3q curvature term for u; the
  faithfulness-decile curves for F. F enters as z-scored Fisher-z
  (atanh, clipped at 1 - 1e-6; `frame.FAITH_CLIP`) -- the standard variance
  stabiliser for correlations, decompressing the near-1 bulk so beta_F is not
  purely tail-driven. Verify against `faithfulness.csv` (the leverage rows
  exist for exactly this); the cached `k_true.npy` / `k_tilde.npy` allow a
  Pearson-based variant without touching the satisfactions tensor.
- **S2 (independence).** Targets are independent generator draws; one row per
  formula, no clustering level exists.
- **S3 (non-collapsibility).** Logistic coefficients grow mechanically when
  outcome-predictive covariates enter, absent any confounding. Rung-to-rung
  comparisons are therefore made on the probability scale
  (`marginal_effects.csv`: average +1 SD g-computation shifts); the log-odds
  attenuation blocks are retained but read jointly with their AME-scale
  counterparts.
- **S4 (first-stage uncertainty vs attenuation).** The bootstrap re-derives
  every generated regressor (bin edges, binned means/SDs, studentisation, the
  Fisher-z standardisation) inside each resample, so first-stage estimation
  uncertainty is inside the intervals. It does *not* de-attenuate: under the
  population-semantics reading, noise in F (256 landmarks; Spearman SE ~ 0.06
  near rho = 0) attenuates beta_F and leaves the F-backdoor for u only
  partially closed. Quantifiable via split-landmark reliability if needed.
- **S5 (coding and cells).** Depth enters as cell means (per-depth absolute
  log-odds; covariate coefficients unchanged). Some depth x operator cells
  are structurally sparse or empty (a binary operator forces depth >= 1), so
  "all else equal" operator contrasts extrapolate there; tiny cells produce
  extreme cell-mean coefficients that do not contaminate other terms.
- **S6 (inference).** HC1 SEs accompany point tables; reported intervals are
  95% percentile-bootstrap CIs from the whole-pipeline bootstrap. The
  confirmatory family is beta_V @ M2, beta_u @ M3, beta_F @ M3, plus the
  comparative joint operator contrasts @ S; it is reported in full with NO
  multiplicity adjustment (the project is declared exploratory and the family
  is small and pre-stated). Everything else is secondary, descriptive, or
  diagnostic (manifest tier map). This family supersedes the earlier
  single-primary declaration (beta_u in M1) -- a design-stage revision made
  on identification grounds before the present pipeline produced numbers,
  recorded in the manifest (``design_revision``).

## 4. The rung lattice (`m_ladder.csv`, `operators.csv`, `marginal_effects.csv`)

Every rung includes C(depth); every downward edge adds exactly one block, so
each cross-rung movement has one interpretation. Q1-Q4 mark where the four
confirmatory readings live; all other printed coefficients are scaffolding.

```
                        M0 : C(depth)                          [baseline]
                             │
        ┌────────────────────┼──────────────────────┐
        ▼                    ▼                      ▼
  S : has_op           M1 : V + u             F1 : V + F       [branch starts:
  [operators.csv,           │ +has_op              │ +has_op    minimal-adjustment
   depth_curve.csv]         ▼                      ▼            associations]
  Q4 lives here        M2 : V + u + S         F2 : V + F + S
                       Q1 lives here               │
                            │ +z_faith             │ +u
                            └─────────┬────────────┘
                                      ▼
                            M3 : V + u + F + S                 [the meet]
                            Q2 + Q3 live here
                                      │
                                      ▼
                            M3q : M3 + u^2                     [curvature check]
```

| Rung | Model (+ C(depth)) | Reading | Rests on |
|---|---|---|---|
| M0 | -- | correctness stratified by depth | S2, S5 |
| S | has_op x 8 | **Q4: operator contrasts** (comparative, total: geometry path open) | A3-adjacent, S2, S5 |
| M1 | V + u | u-branch start: minimal-adjustment association | S1, S2 |
| M2 | V + u + S | **Q1: total effect of V** (beta_V) | A2, A3, S1-S6 |
| F1 | V + F | F-branch start: minimal-adjustment association | S1, S2 |
| F2 | V + F + S | F-branch syntax-absorption step | S1, S2 |
| M3 | V + u + F + S | **Q2: beta_F; Q3: beta_u** (+ direct-V side-reading) | A1, A3, S1-S6 |
| M3q | M3 + u^2 | curvature check at the Q3 rung | S1 check |

Per-rung notes -- what may and may not be said:

- **M0.** Descriptive stratification. No causal depth reading: the S <-> D
  edge leaves the operator mix uncontrolled, and depth admits no well-defined
  intervention. Baseline, nothing more.
- **S rung.** The joint operator model; its table is `operators.csv` and its
  standardised depth profile is `depth_curve.csv` (Section 5). Its contrasts
  are the fourth member of the confirmatory family, in the *comparative*
  sense only.
- **M1 and F1 (branch starts).** Minimal-adjustment associations; under the
  DAG neither coefficient is causal (open backdoors through omitted S, and
  through U_geo via the omitted other geometry coarsening). Their role is to
  anchor the attribution trajectories, not to estimate effects. Do not let
  causal vocabulary attach to them.
- **M2.** Two readings, keep them separate. (a) **Q1, confirmatory** -- total
  effect of V: backdoors V <- S -> C and V <- D -> C closed; F deliberately
  excluded, because it is a mediator of V (V -> F -> C belongs to the total
  effect, by the directed reliability channel C4) and because the
  unconditioned collider at F blocks every U_geo path, making this the most
  assumption-robust estimand in the experiment (needs A2 + A3, survives a
  failure of A1). u stays in the model uninterpreted: by C1 it is no
  descendant of V, so it cannot disturb the reading; it adds precision and
  makes M2 -> M3 a single-term step. (b) *Secondary*: the syntax-absorption
  step of the u trajectory.
  - *Contingency on the V -> F edge (C4).* The "total effect" reading assumes
    V -> F is directed (F a mediator). Were the edge instead purely
    bidirected (V <-> F via a latent), F would be a latent-confounded
    correlate, no rung would identify V's total effect (M2 over-counts the
    confounded correlation; conditioning F only partially adjusts, F being a
    noisy proxy), and Q1 would downgrade to a syntax-adjusted *association*.
    Only Q1 is affected: Q2/Q3/Q4 condition on V regardless of why V and F
    co-move, so they are invariant to the edge's direction.
  - *The design measures the disputed flow -- no a priori resolution needed.*
    The quantity in dispute is how much of V's association with C runs through
    F, which is exactly the **M2 -> M3 movement of beta_V** (F excluded ->
    conditioned). If that movement is ~0 (and `faith_grid.csv` shows F roughly
    flat across variance cells), no meaningful flow crosses the edge,
    M2 ~ M3 for beta_V, and Q1 is robust whichever way the edge points. If it
    is substantial, the directed reading is the one that pays rent and is
    defended by C4. Report Q1 as "total effect" or as "association"
    accordingly.
- **F2.** The syntax-absorption step of the F trajectory; beta_F here is
  still not causal (u <- U_geo backdoor open until u enters at M3).
- **M3 (the meet).** The minimal model in which beta_u and beta_F are
  interpretable, and they license *each other*: F needs {S, D, V, u} in the
  model (u blocks F <- U_geo -> u -> C), u needs {S, D, F} (F blocks
  u <- U_geo -> F -> C). They enter together or not at all; adding F to a
  model without S would instead open collider paths through the omitted
  syntax block. Readings:
  - **Q2, beta_F**: total = direct (F has no measured descendants); causal
    under A1 + A3 + S1.
  - **Q3, beta_u**: total = direct (C2: no u -> F edge); causal under
    A1 + A3 + S1.
  - *Side-reading, beta_V*: the direct effect of V net of measured
    faithfulness (the collider at F is re-blocked through S, D, u, leaving
    A1 as the exposure). V's total effect lives in M2, not here; the
    M2-vs-M3 difference in V's AME is the share transmitted through
    measured F (mediation-flavoured; requires A1).
- **M3q.** Exploratory curvature check for u, run at the rung where u is
  interpreted (checking functional form on a model whose coefficient is not
  read would check the wrong thing); motivated by the u-decile curve.

**Attribution trajectories (secondary; `m_ladder.csv` attenuation rows,
AME scale in `marginal_effects.csv`):** u along M1 -> M2 -> M3 (syntax
share, then shared-latent share through measured F); z_faith along
F1 -> F2 -> M3 (syntax share, then scale share through u); z_variance along
M1 -> M2 -> M3 (compositional step, then mediation-through-F share). An
attenuation sequence requires an ordering and no single nested chain can
watch both u and F from the start -- hence the two branches meeting at M3.

**`marginal_effects.csv`.** Average predicted-probability change for a +1 SD
shift, g-computed over the observed covariate distribution, per rung; plus
AME-scale trajectory rows. Same estimand status as the coefficient each row
accompanies; this is the scale on which rungs are *compared* (S3).

## 5. Operator and depth exhibits (the S rung)

- **`operators.csv` (H2c).** The headline column is the joint model
  `correct ~ has_op x 8 + C(depth)`: each coefficient is operator presence at
  matched depth *and* matched co-occurring operators, with `gap_joint` its
  0 -> 1 marginally standardised probability companion. `log_odds_single`
  (one operator at a time + depth) is retained as the co-occurrence-
  confounded companion; the movement between the two columns *exhibits* that
  confounding. Geometry covariates are deliberately excluded: V, u, F are
  mediators of S, so these are total contrasts with the geometry channel
  included. Claims are comparative and associational ("formulas containing U
  show lower adjusted correctness"), never interventional (no well-defined
  syntax intervention; A3-type residual traits such as size at fixed depth
  remain unadjusted; sparse S x D cells extrapolate, S5). This is a
  decomposition exhibit, not eight hypotheses.
- **`depth_curve.csv`.** Raw correctness per depth plus the operator-
  standardised profile from the joint fit (each target keeps its operator
  profile, its depth cell is forced to d, predictions averaged). The
  standardised variant licenses "at matched operator mix" sentences only;
  depth remains the coarsest complexity proxy and gets no causal reading.

## 6. Descriptive curves

Curves are companions to the confirmatory readings, not rungs, and
deliberately carry *lighter* adjustment than the rung they accompany: a
descriptive exhibit should not lean on A1. Each of V, u, F therefore has the
pair {decile curve, coefficient trajectory}.

- **`var_curve.csv` (variance deciles; Q1 companion).** Raw and
  depth-adjusted correctness by variance decile -- the shape exhibit behind
  beta_V (and the exhibit in which any degenerate-limit floor behaviour
  would show).
- **`curve.csv` (u deciles; Q3 companion).** Depth-adjusted via marginal
  standardisation. Because u is variance-residualised by construction (C1),
  this curve is implicitly depth+variance-adjusted. Motivates M3q.
- **`faith_curve.csv` (faithfulness deciles; Q2 companion).** Two variants:
  `adj_rate` (depth-adjusted) and `adj_rate_vd` (depth+variance-adjusted,
  via variance-decile indicators). The vd variant is the matched pair to the
  u curve: both are then net of depth and variance, and the two can be read
  side by side as the two marginal geometry gradients (scale vs direction).
- **Cross-adjustment rule.** The u curve is never adjusted for F and the F
  curve never for u. Both are coarsenings of the same vector (C2);
  conditioning one while displaying the other induces selection distortion
  through U_geo -- a distributional phenomenon that applies to descriptive
  exhibits just as to effect estimates. The F-netted u gradient needs no
  curve: that object *is* beta_u @ M3.

## 7. Stage A exhibits and checks

- **`occupancy.csv`** -- motivation: joint spread of variance and raw norm.
- **`norm_variance.csv`** -- the binned E[log norm | variance] curve that u
  residualises against; documents C1 empirically.
- **`faithfulness.csv`** -- distribution + leverage stats; the S1 check for
  the Fisher-z choice (inspect before defending beta_F's functional form).
- **`faith_grid.csv`** -- mean F over variance x u cells; the descriptive
  face of the U_geo dependence (C2). Its variance margin, E[F | variance], is
  also the arbiter for the V -> F edge (C4): a flat margin means the edge
  carries ~no flow and Q1 is robust to the edge's direction.
- **`diagnostic.csv`** -- R^2 of u on operator features; certifies that u is
  not operator-determined, i.e. M2/M3 retain within-syntax variation in u
  (an anti-collinearity certificate for the ladder, computed without
  outcome data).
- **`op_signature.csv`** -- mean u by operator presence; the covariate-side
  bridge between the syntax and geometry blocks.
- **`shuffle_null.csv`** -- falsification check: equivalence under shuffled
  embeddings at chance, flat in variance. Certifies the information
  bottleneck (partial discharge of A1) and kills the mechanical-guessability
  confound for V (supports A2).

## 8. Threats and optional probes (quantify, do not hand-wave)

| Threat | Hits | Probe (all post-hoc, label as robustness) |
|---|---|---|
| A1: residual embedding structure -> C | beta_F, beta_u, direct beta_V (M3) | add top PCA components of the embeddings to M3; report movement of beta_F, beta_u |
| A2: latent variance-difficulty trait | total beta_V (M2) | add richer syntax features (operator counts, size); report movement of beta_V |
| C4: V -> F bidirected not directed | total-effect reading of beta_V (M2, Q1 only) | read the M2 -> M3 movement of beta_V and the E[F\|variance] margin of `faith_grid.csv`; if both ~flat, Q1 robust either way; else defend the reliability channel and label the measurement-artifact share |
| A3: size/nesting at fixed depth | all rungs >= M2, operators.csv | operator counts instead of presence; formula length as extra adjuster |
| S1: functional form of F | beta_F (M3) | Pearson variant from cached `k_true.npy` / `k_tilde.npy`; spline in z_faith; compare with faith-decile curves |
| S4: landmark noise in F | beta_F attenuation; partial u backdoor | split-landmark reliability; simple correction factor |
| S5: sparse S x D cells | operators.csv contrasts | report cell occupancy alongside the contrasts |

## 9. Suggested defence order for the chapter

1. Units and constructional facts (C1-C5): what is *built*, not assumed --
   the variance-orthogonality of u, the scale/direction split, and the
   computational priority of syntax (C5) are the chapter's firmest ground;
   lead with them. C4 is the one hybrid: the scale half is constructional,
   the reliability half is a defended mechanism.
2. The DAG: the syntax roots S, D shown computationally prior (C5) so the
   S, D -> geometry edges are oriented by construction, not assumed; the two
   dashed edges read as latent projections; the one directed geometry edge
   (V -> F) defended by the reliability channel (C4) with its scale half shown
   zero; then the absence audit: A1-A3 stated as the price of each causal
   reading, with C5 (orientation, certain) kept distinct from A3
   (sufficiency, approximate).
3. The lattice (Section 4): one rung per question, not one model for
   everything -- total-V (M2) and F/u (M3) are mutually exclusive within a
   single regression; the branches exist because an attenuation sequence
   requires an ordering and u and F cannot both be watched from the start of
   one chain. The lattice is an estimand structure, not model selection.
4. The confirmatory family (beta_V @ M2, beta_u @ M3, beta_F @ M3, operator
   contrasts @ S) reported in full with CIs and no multiplicity adjustment,
   with the design-revision note stated once; causal vocabulary confined to
   the family sentences with their assumptions named inline; branch-start
   coefficients kept strictly associational.
5. Exhibits: operators and depth as adjusted comparisons; the three decile
   curves as companions with the cross-adjustment rule and the matched-pair
   symmetry.
6. Limitations with numbers attached (Section 8 probes), not apologies.
