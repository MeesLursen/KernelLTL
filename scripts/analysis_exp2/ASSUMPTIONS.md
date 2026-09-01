# Experiment 2: estimands, adjustment sets, and the assumptions each output rests on

Companion to `run_exp2.py` and the diagram (`misc/Causal_diagram_EXP2_V2.png`).
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

- **C1 (u is variance-orthogonal).** u = (log10||emb|| - binned mean) / binned
  SD within 50 variance-quantile bins removes the mechanical Cauchy-Schwarz
  dependence of the norm on variance in location and scale. The missing
  V -> u edge is enforced, not assumed. Realised Spearman(u, V) = +0.028.
  *Edge qualification (do not state C1 unqualified).* The bins are equal-COUNT,
  not equal-width, and the low-variance stratum is sparse: bin 0 spans 2.21
  decades of variance against a median bin width of 0.016, and bins 0-1 hold
  4% of targets. Inside those two bins "within-bin" does not mean "at matched
  variance" -- within bin 0, log10||emb|| regresses on log10 V with R^2 = 0.994
  and u retains Spearman +0.99 with variance. So for 4% of targets u does not
  measure what its name says; this is a CONSTRUCT-VALIDITY limit on the label,
  not an estimation error (u is a defined functional, C3). Bounded: after
  studentisation bin 0 holds 2.0% of u's leverage, exactly its share of the
  sample, and the extreme-u tail is NOT concentrated there (2 of the ~50
  targets with u < -3). Quantified in Section 8.
  *Base.* log10 rather than natural log, so every logged column and figure axis
  reads in DECADES, matching the log-scaled variance axis beside it. u is a
  ratio of logs and therefore exactly base-invariant, as is every coefficient;
  only `norm_variance.csv`'s logged columns depend on the choice.
  *What the construction buys beyond a missing edge -- and it is more than
  "uncorrelated".* Within-bin centring and studentisation force
  E[u | bin] = 0 and E[u^2 | bin] = 1 EXACTLY, in every bin. Since the bins are
  narrow in V, that is mean-independence, not merely zero correlation, and it
  has a consequence that matters once S1 is taken seriously: **u's curve is
  immune to any misspecification of V, and V's curve to any misspecification of
  u.** Mean-independence gives cov(u, g(V)) ~ 0 for EVERY function g -- linear,
  quadratic, decile indicators alike -- and E[u^2 | bin] being constant across
  bins gives the symmetric statement for u's leading nonlinear term. So when
  linearity failed for both (S1), only F's readings were exposed to residual
  confounding, because F has no such construction behind it and correlates
  +0.354 with V and +0.294 with u. Predicted from the construction, then
  measured: re-fitting the curves with adequate adjusters moves V's by 0.32 pp
  on average and u's by 0.21 pp, against 0.98 pp for F's, whose amplitude falls
  by 39%. The immunity is exact only insofar as the binning genuinely holds V
  fixed, so it is weakest in bins 0-1 -- the same 4% of targets the edge
  qualification above already flags. Also the reason `vif_u_with_z_variance` is
  1.0000 in `norm_variance_stats.csv` where the raw norm beside variance gives
  12.94, and 63.80 on the log scale.
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
  *And the empirical arbitration is now weaker than the earlier draft claimed.*
  Under the adequate specification, conditioning on F moves the V CURVE by
  0.46 pp on average with no bin's paired interval excluding zero. The
  "55% through measured F" figure came from the linear beta_V halving between
  rungs, and that coefficient was a straight line through a hump. C4's
  mechanism argument stands on its own terms (Section 2, and the reliability
  evidence below); what does not stand is the claim that the lattice measured a
  large flow through F. Also note the caveat's own escape clause has now fired:
  beta_F's interval includes zero at M4, so "if beta_F is null the caveat is
  moot" is live rather than hypothetical.
  *C4 surfaces three times, and it is ONE mechanism each time -- say it once
  properly rather than three times in passing.* (i) F's LEVEL tracks
  measurement precision across variance strata: mean z_faith is -0.533 /
  +0.369 / +0.164 for low / mid / high variance against a noise proxy of
  0.133 / 0.068 / 0.079 -- inverse in ordering including the non-obvious part,
  since F peaks at MID variance, which is also where the noise floor is lowest
  (`faith_by_variance.csv`). (ii) F's SPREAD shrinks the same way (SD 0.831 /
  0.881 / 1.042). (iii) F's RESPONSIVENESS to u shrinks the same way: the
  low->mid u-tercile step is +0.335 / +0.570 / +0.712, the low-variance value
  smaller than the other two by 2.8 and 4.0 SE. Rank noise attenuates a
  Spearman estimate toward zero and therefore shrinks level, spread and
  responsiveness TOGETHER; all three move as predicted. Three strata is
  suggestive rather than decisive, but the correspondence is exact in ordering.
  The same mechanism produces the positivity hole in S7. The concession this
  forces: part of "low-variance targets are less faithful" is "faithfulness is
  harder to MEASURE at low variance", and it belongs in the results with a
  number attached rather than in a limitations list as a hedge.
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
  *Consequence: the two channels are ENTANGLED, so their magnitudes are not
  comparable.* Because the geometry covariates are computed downstream of the
  operator set, syntax moves all three of them, and by comparable amounts
  (`diagnostic.csv`): R^2 on has_op is 0.123 for V, 0.119 for u, 0.154 for F,
  rising only to 0.126 / 0.133 / 0.158 with depth added. Two readings follow.
  (i) IDENTIFICATION IS SAFE: 84-88% of each covariate's variance survives
  operator adjustment, so there is ample within-syntax variation to read a
  coefficient from. (ii) The M1 operator contrasts carry NO geometry term, so
  a contrast of -11.9 pp for G already contains whatever G does by shifting V,
  u and F -- which `op_signature.csv` puts at -0.28, -0.35 and -0.31 SD
  respectively. The operator contrasts and the geometry readings are therefore
  **not additive and not rankable against each other**, and any sentence that
  compares their sizes needs this stated first. The same entanglement is why
  `adequacy.csv` reports the AUC decomposition in BOTH orders: the shared
  portion is a reported quantity rather than an artefact of whichever block is
  entered first.
  *What is NOT supported.* The natural reading -- that costly operators are
  costly BECAUSE the kernel maps them badly -- does not survive testing.
  Adjusting for all three geometry covariates leaves the operator contrasts
  essentially where they were (mean |shift| 0.90 pp on contrasts of 2.5-12 pp;
  only F(eventually) moves by as much as a quarter of its contrast, and three
  of eight move away from zero). Across the eight operators the correlation
  between geometry depression and correctness cost is null. This is a
  between-operator statement and does not touch the within-operator geometry
  readings, which are estimated with has_op adjusted.

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

- **S1 (functional form). REJECTED FOR TWO OF THE THREE COVARIATES, and the
  models were respecified rather than caveated.** This is the largest revision
  in the file and it is a result, not a technicality.

  *The test.* Box-Tidwell is unavailable here and not merely awkward: it adds
  x*ln(x) and so requires x > 0, while all three covariates are centred and
  take negative values -- and u, a signed residual by construction, has no
  positive scale at any point, so the covariate whose form matters most is the
  one Box-Tidwell can never reach. The replacement is the grouped lack-of-fit
  test (Hosmer-Lemeshow, *Applied Logistic Regression* 4.2.1): swap the term
  for decile indicators, compare by likelihood ratio on 8 df. It is
  sign-agnostic and assumes no functional form for the alternative. Over a
  D + S base (`spec_search.csv`):

  | covariate | LR | p | |
  |---|---|---|---|
  | z_variance | 23.43 | 2.9e-3 | **rejected** |
  | u | 37.76 | 8.3e-6 | **rejected** |
  | z_faith | 8.22 | 0.41 | holds |

  *The forms adopted.* V enters as a quadratic (captures 78% of the departure;
  residual against a decile reference p = 0.649). u enters as NINE DECILE
  INDICATORS: the quadratic captures only 49% and its residual survives
  (p = 0.0072), while deciles are not improved on by a 20-bin cut (p = 0.41).
  F stays linear.

  *The search is complete over monotone reparametrisations, and that is why no
  log-V or entropy variant is tested.* Quantile binning is invariant to
  monotone transforms, so the decile model already achieves whatever the best
  monotone rescaling of V could achieve at that resolution -- there is nothing
  left for log V to find. Confirmed directly for the natural candidate: binary
  entropy H(p) is rank-identical to V (Spearman = 1.000000, identical deciles
  for all 4000 targets) and does not rescue linearity (LR 20.84, p = 0.0076).

  *What is NOT repaired.* V is adequate only AT DECILE RESOLUTION. Against a
  20-bin reference the quadratic fails (p = 0.035), and so do the cubic
  (p = 0.043) and the deciles themselves (p = 0.008). V carries structure below
  decile width -- most plausibly at the p(1-p) ceiling, where decile 9 spans
  0.6% of the range -- and one polynomial order higher does not reach it. Every
  "adequately specified" number in this analysis inherits that.

  *Consequence: V and u lose their scalar readings.* An average marginal effect
  is the mean vertical displacement from sliding every target one SD along the
  response curve; on an inverted U it averages positive displacements left of
  the peak against negative ones right of it. V's +1 SD AME is **-3.71 pp** at
  M2 against **+1.18 pp** under the rejected linear form -- both defensible
  summaries of one curve, disagreeing in sign, because the quantity tracks
  where the population sits relative to the optimum rather than the strength of
  the relationship. Under decile coding for u it is not even defined. The
  CURVES are the estimands for Q1 and Q3; only F carries a scalar.

  *The substantive reading.* Both quantities were introduced as monotone goods
  -- more variance means the satisfaction vector says more, higher u means the
  anchor set is not under-representing the embedding -- and both have an
  optimum. High V means p near 0.5, so "most informative" and "least
  determinate" are the same region; and over-exposure costs about as much as
  under-exposure. The direction quantity has no optimum. That contrast is the
  chapter's finding.

  *F's transform is unchanged and its caveat stands.* F enters as z-scored
  Fisher-z (atanh, clipped at 1 - 1e-6; `frame.FAITH_CLIP`). On raw rho the
  bottom 5% of targets carry 59.2% of the squared-deviation leverage (skew
  -2.86); on the Fisher-z scale 31.0% (skew -0.55), against a uniform reference
  of 13.6% and a normal one of 21.9% (`covariates.csv`). *Caveat:* atanh is
  DERIVED as the variance stabiliser for a PEARSON correlation
  (Var(r) ~ (1-rho^2)^2/n, so h' ~ 1/(1-rho^2) integrates to atanh). F is a
  SPEARMAN correlation, whose stabilised variance is ~1.06/(n-3) rather than
  1/(n-3); atanh is therefore a well-motivated monotone decompressor with an
  approximate stabilising property, not an exact one. The cached `k_true.npy` /
  `k_tilde.npy` allow a Pearson-based variant without touching the
  satisfactions tensor.

  *Provenance.* The search was run after the decile curves were seen. It
  selected on shape adequacy and not on any estimand's value, and it made the
  headline result WEAKER -- beta_F's AME roughly halves once its adjusters are
  adequately specified -- which is the evidence that it was not outcome-driven.
  The rejected linear fits stay in `m_ladder.csv` (rows `L-M2`, `L-M4`) so the
  search can be audited rather than taken on trust. See also S6 on
  post-selection inference.
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
  *The symmetric statement for u, which the earlier draft omitted.* Finite-N
  noise afflicts the norm as well as the correlation, so u has the analogous
  problem in a different form. It does NOT produce a V -> u edge: targets in a
  bin share a variance hence a noise floor, so the systematic component is
  absorbed into the bin mean by construction (C1). What survives is
  DIFFERENTIAL RELIABILITY -- noise is a larger fraction of the signal where
  the true norm is smallest, so u is noisier at low variance. That is
  heteroskedastic measurement error in a covariate, not an arrow; decorrelation
  by construction removes the edge but not the reliability gradient. Direction
  check: a pure noise floor would INFLATE small norms and flatten the low-V end
  of the ridge, whereas the observed low-tercile slope is 0.846 against 0.5 for
  pure scale -- so noise is masking part of the decay and the true
  under-registration is if anything steeper than the fitted 0.814.
  *What the bootstrap cannot cover.* It measures sampling variability
  CONDITIONAL ON THE ESTIMATOR. Every resample uses 50 equal-count bins, so
  every resample shares the C1 edge qualification identically; that limitation
  contributes nothing to interval width and must be reported separately
  (Section 8), not assumed to be inside the CIs.
- **S5 (coding and cells).** Depth enters as cell means (per-depth absolute
  log-odds; covariate coefficients unchanged). Some depth x operator cells
  are structurally sparse or empty (a binary operator forces depth >= 1), so
  "all else equal" operator contrasts extrapolate there; tiny cells produce
  extreme cell-mean coefficients that do not contaminate other terms.
- **S7 (positivity: a hole the design cannot fill).** High F never co-occurs
  with low V. In the decile cross-tab (`occupancy.csv`, rows `z_variance` x
  cols `z_faith`) three cells are EMPTY and eight hold fewer than ten targets,
  all in the high-F / low-V corner. So when the F curve forces F to its top
  decile for every target, the 800 targets in V-deciles 0-1 receive a
  prediction for a combination that occurs zero times -- **20% of the top
  decile's standardisation weight has no local support**, 10% for the ninth.
  This is a property of the DESIGN, not of the estimator: it afflicts any
  functional form, and the linear specification extrapolated into the same hole
  more freely, merely without making it countable. And by C4 it is structural
  rather than a sampling accident -- low variance means a noisy Spearman, so
  high F is not observable there and more data will not fill it in. Distinct
  from S5, which concerns S x D cells.
- **S6 (inference).** HC1 SEs accompany point tables; reported intervals are
  95% percentile-bootstrap CIs from the whole-pipeline bootstrap. The
  confirmatory family is the V curve @ M2, the u curve @ M4, beta_F @ M4 with
  its AME, plus the comparative joint operator contrasts @ M1; it is reported
  in full with NO multiplicity adjustment (the project is declared exploratory
  and the family is small and pre-stated). Everything else is secondary,
  descriptive, or diagnostic (manifest tier map). This family supersedes the
  earlier single-primary declaration (beta_u in M1) and, at revision v3, the
  scalar readings for V and u (S1).
  *Post-selection inference.* The intervals are CONDITIONAL ON THE SELECTED
  SPECIFICATION and do not account for the search that chose it (S1). A fully
  honest interval would re-run the selection inside every bootstrap resample;
  this one does not. The search is reported in full (`spec_search.csv`) and the
  rejected fits are tabulated beside it, so a reader can price the omission
  rather than having to trust it. Two facts bound how much it can matter: the
  selection criterion was shape adequacy rather than any estimand's value, and
  the selected specification made the headline result weaker.

## 4. The rung lattice (`m_ladder.csv`, `operators.csv`, `marginal_effects.csv`)

Every rung includes C(depth); every downward edge adds exactly one block, so
each cross-rung movement has one interpretation. V enters as `z_variance +
z_variance_sq` and u as nine decile indicators throughout (S1).

**Syntax sits at the BASE, not in an adjustment step.** C5 makes S and D pure
functions of the formula string, fixed before any trace is drawn, so a rung
that reads a geometry quantity with the operator set left open is not an
estimand anyone would report -- and operators shift all three geometry
covariates by 12-16% of their variance (`diagnostic.csv`). The lattice then
forks symmetrically at M2.

```
                    M0 : C(depth)                     [baseline]
                     │ + has_op
                     ▼
                    M1 : + S                          Q4 lives here
                     │ + V + V^2                       [operators.csv,
                     ▼                                  depth_curve.csv]
                    M2 : + V                          Q1's rung
                    ╱ ╲
            + u    ╱   ╲   + z_faith
                  ▼     ▼
                M3u     M3F
                  ╲     ╱
            + F    ╲   ╱   + u
                    ▼
                    M4 : + V + u + F                  Q2 + Q3's rung [the meet]
```

**What the fork buys, and what it costs.** It makes M3u -> M4 ("what does F do
to u") and M3F -> M4 ("what does u do to F") the same kind of step, which the
old asymmetric M1/M2/M3-versus-F1/F2 chain could not. It also gives V two
single-block steps from one base: **M2 -> M3u is what u ALONE does to V** and
**M2 -> M3F is what F alone does**. The old chain could not produce that pair
because u already sat inside M2.
*Measured, and it does not go the way C4 leads one to expect.* Adding u moves
the V curve by 0.28 pp on average (max 0.92); adding F moves it by 0.46 pp
(max 1.18). F's is the larger, as C4 predicts, but only by a factor of 1.6 --
and NO bin's paired across-step interval excludes zero for either. So the
V curve is close to insensitive to both, and the earlier "55% of beta_V runs
through measured F" was substantially an artefact of summarising a curved
relationship with a straight line: the linear coefficient fell by half between
those rungs while the curve itself barely moved. C4's directed component
survives as a mechanism argument (Section 2) but its empirical support here is
much weaker than the linear attenuation suggested. Report the curve movement,
not the coefficient attenuation.
The cost is that with syntax at the base there is no `+S` step to attenuate
along; that comparison survives in the CURVE sequences (Section 6), which
begin at raw and pass through D + S. Curves are descriptive, so their sequence
may include steps the lattice does not.

**M2 -> M3F is a step too far for Q1.** F is computed downstream of V (C4), so
conditioning on it changes what the V curve refers to. It is computed only to
price the decision not to read there.

**Provenance (disclose once, in the text).** The specification search that
chose these forms ran after the decile curves were seen; it selected on shape
adequacy rather than on any estimand's value, and it made the headline result
weaker. M2q and M3q are retired as rungs -- they are now rows of
`spec_search.csv`. The rejected linear fits stay in `m_ladder.csv` as `L-M2`
and `L-M4` so the search is auditable. See S1 and S6.

| Rung | Model (+ C(depth)) | Reading | Rests on |
|---|---|---|---|
| M0 | -- | correctness stratified by depth | S2, S5 |
| M1 | S | **Q4: operator contrasts** (comparative, TOTAL: geometry path open) | C5, S2, S5 |
| M2 | + V + V^2 | **Q1: the V CURVE** (no scalar -- S1) | C4, C5, S1-S7 |
| M3u | + u | what u alone does to V; C1's prediction, tested | C1, S1-S6 |
| M3F | + F | what F alone does to V; a step too far for Q1 | C4, S1-S6 |
| M4 | + V + u + F | **Q2: beta_F + its AME; Q3: the u CURVE** | C2, C5, S1-S7 |
| L-M2, L-M4 | linear V, linear u | **REJECTED** (tabulated for audit only) | -- |

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
  failure of A1). u stays in the model uninterpreted, and for exactly one
  reason: it makes M2 -> M3 a single-block step, so that step's movement in
  beta_V is attributable to z_faith alone -- and that movement is the C4
  arbiter. It buys nothing else. Dropping u moves beta_V from +0.0972 to
  +0.0939 (AME +1.23 -> +1.18 pp), i.e. nothing, because by C1 it is no
  descendant of V (corr = +0.004); adding F instead moves it to +0.0558, a 43%
  shift, because F is. That contrast is C1-versus-C4 measured rather than
  argued. It does NOT add precision: HC1 SE 0.0483 -> 0.0484, z-statistic
  2.01 -> 1.94 -- expected under S3, since an outcome-predictive covariate
  inflates a logistic coefficient and its SE together. (Earlier drafts of this
  file and of `models.py` asserted a precision gain; it was never checked and
  is not there.) (b) *Secondary*: the syntax-absorption step of the u
  trajectory.
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
- **M2q.** Exploratory curvature check for V, run at the rung where beta_V is
  interpreted; motivated by the variance-decile curve, which rises 0.267 ->
  0.430 across deciles 1-5 and falls back to 0.363. V^2 = -0.280 (z = -4.30),
  the linear term collapses to +0.020, and the turning point sits at +0.04 SD
  -- essentially at mean variance. Read together with M2: the confirmatory
  linear beta_V is a straight line through a near-symmetric hump, which is
  why it is small. Do NOT restate Q1's estimand as the quadratic one.
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

*Report the delta, never the ratio.* Both are stored, but the ratio is a
quotient of two random quantities and is unusable whenever the lower rung
sits near zero. beta_u @ M1 is -0.055, which gives AME-scale ratio intervals
of [-10.04, +8.65] (M1 -> M2) and [-4.85, +3.94] (M2 -> M3); even V's
M2 -> M3 ratio runs to [-0.02, +3.62]. Every delta, by contrast, has a
percentile interval narrower than 1.1 pp. Only F1 -> F2 (+0.40 [+0.22,
+0.75]) has a ratio worth quoting, and no ratio may appear without its CI.
The delta is also what fig07 plots, so a reader can verify it off the axis.
Sign convention: delta = a - b tracks the direction of the change, not
distance from zero -- multiply by sign(a) before calling a movement
shrinkage, or a negative coefficient growing more negative reads as
attenuation.

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

## 6. Descriptive curves -- and for V and u, THE ESTIMANDS

Because linearity is rejected for V and u and no scalar summarises a
non-monotone relationship honestly (S1), the curves are not companions to a
coefficient for those two: they ARE Q1's and Q3's readings. F keeps a
coefficient, and its curve is a companion in the older sense.

**Sequences, not single curves.** Each covariate's curve is reported at a
sequence of adjustment sets (`models.CURVE_SEQ`), and one step of each sequence
is marked `primary_step` -- the rung whose estimand it is. The earlier steps
are the attenuation sequence, which is how the adjustment story is told now
that no scalar exists to attenuate.

| file | sequence | primary |
|---|---|---|
| `curve_z_variance.csv` | raw, D+S, +u, +F | **D+S** (= M2) |
| `curve_u.csv` | raw, D+S, +V, +V+F | **D+V+F** (= M4) |
| `curve_z_faith.csv` | raw, D+S, +V, +V+u | **D+V+u** (= M4) |

The `raw` step is the empirical per-bin rate and carries no model at all. Note
that the sequences include a D+S step the LATTICE no longer has (syntax sits at
its base, Section 4): curves are descriptive, so their sequence may include
rungs that are not estimands, and this is where the syntax-absorption
comparison survives.

**Binned on the MODEL scale.** Bins are cut on `z_variance`, `u` and `z_faith`
rather than on raw variance and rho. For V this is an affine relabel and
changes nothing; for F it removes a Jensen gap, since atanh is convex so
atanh(mean rho) != mean(atanh rho) and the first decile's plotted position was
off by 2.3% of the axis. Raw-unit means are kept alongside (`mean_variance`,
`mean_relational_faithfulness`) so a figure can label its axis interpretably
while the bin positions stay on the scale the models use.

**Paired across-step differences carry their own intervals.** Every non-primary
step also stores `vs_primary_*` with a bootstrap CI. Comparing one bin across
two steps is a PAIRED contrast on the same targets, so its interval is far
tighter than either step's marginal one; quoting the marginals against an
across-step movement would understate the evidence rather than overstate it.

**A step too far, priced rather than hidden.** `curve_z_variance.csv`'s `+F`
step conditions on a quantity computed downstream of V (C4). It is not Q1's
reading and is computed only to show what reading it there would have cost.

*Why the old cross-adjustment prohibition is gone.* F is a collider
(V -> F <- U_geo, and S, D -> F), so conditioning a u curve on F ALONE opens
U_geo <-> V and U_geo <-> S. That is a real hazard of PARTIAL cross-adjustment;
at the full set those paths are blocked by S and D. Empirically it is small
here -- adding F to the u curve moves it by <= 0.010, adding u to the variance
curve by <= 0.002, itself a confirmation of C1 -- so the rule is stated with its
magnitude rather than as a load-bearing caveat.

*Retired:* the `adj_rate_vd` faithfulness variant, and the earlier
"curves carry lighter adjustment" scheme whose stated reason (that a
descriptive exhibit should not lean on A1) was a category error -- A1 is
GEOMETRY sufficiency, whereas conditioning on S invokes A3.

*Disclosure.* The sequence structure and the move of the primary step postdate
seeing the curves, and follow directly from S1's rejection. State it alongside
the specification-search disclosure, not separately.
## 7. Stage A exhibits and checks

- **`occupancy.csv`** -- TWO decile cross-tabs doing two different jobs.
  `variance` x `emb_norm` MOTIVATES A CONSTRUCTION: 49 of 100 cells empty, the
  corners unpopulated, so the design supplies no norm contrast at fixed
  variance and u had to be built to manufacture one. `z_variance` x `z_faith`
  LIMITS A CONCLUSION: 3 empty cells in the high-F/low-V corner, which is S7.
- **`norm_variance.csv`** -- the binned E[log10 norm | variance] curve that u
  residualises against; documents C1 empirically.
- **`norm_variance_stats.csv`** -- its consequences. The ridge fit (slope
  0.814 against 0.5 for pure scale, R^2 = 0.984) and the VIF pair that is the
  actual exhibit: entering the raw norm beside variance gives 12.94, on the log
  scale 63.80, and the construction takes it to exactly 1.00.
- **`covariates.csv`** -- shape and leverage for every covariate BEFORE and
  AFTER its transform, with uniform and normal reference rows so the numbers
  interpret themselves. This is the S1 evidence for both transform choices in
  one place: raw rho 59.2% and log10 norm 55.6% of the leverage in their bottom
  5%, against 31.0% and 39.7% after, and V at 13.9% against a uniform reference
  of 13.6% -- which is why V needed no transform at all.
- **`faith_by_variance.csv`** -- C4's reliability channel: F's level, spread
  and u-responsiveness by variance stratum, against the noise proxy.
- **`diagnostic.csv`** -- R^2 of EACH geometry covariate on operator features
  (0.123 / 0.119 / 0.154). Two jobs, per C5: it certifies identification
  (84-88% of each covariate's variance survives operator adjustment) and it
  quantifies the entanglement that makes the operator contrasts and the
  geometry readings non-comparable.
- **`op_signature.csv`** -- JOINT per-operator shift in V, u and F, in SD
  units. Joint rather than marginal because operators co-occur and the
  correctness contrasts these are set beside are themselves joint.
- **`depth_op_mix.csv`** -- operator prevalence per depth cell; the mechanism
  behind the depth curve's operator standardisation.
- **`spec_search.csv`** -- the linearity ladder that chose the specification
  (S1). Methodological rather than Stage A, but listed here because it is read
  before any estimand.
- **`adequacy.csv`** -- AUC by nested block in both orders, Pregibon link test,
  dfbeta influence. Calibration is deliberately absent: maximum likelihood
  solves X'(y - p) = 0 per design column, so predicted and observed match
  exactly within every depth cell and operator group by construction, and
  where calibration CAN fail it is algebraically the same quantity as the
  decile curve's departure from the fitted form, which `spec_search.csv`
  already tests with more power.
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
| S1: non-monotone V and u | beta_V (M2), beta_u (M3) | the two curvature rungs: V^2 @ M2q = -0.280 (z = -4.30), u^2 @ M3q = -0.144 (z = -3.25); both linear readings understate a hump |
| C1: bin width at the sparse low-variance edge | beta_u (M3), the u-decile curve's left arm | *(i)* bin-count sensitivity, below; *(ii)* within-bin linear detrending instead of mean-subtraction fixes bin 0 (Spearman +0.99 -> +0.16) and moves beta_u -0.130 -> -0.115, AME -1.60 -> -1.42 pp -- well inside the CI, so reported, NOT adopted |
| n_bins = 50 is a design constant | beta_u (M3) | across 10/25/50/100/200/400 bins beta_u ranges -0.073 to -0.130, entirely inside the sampling CI [-0.212, +0.002]; disclose that the pre-specified 50 yields the LARGEST magnitude of the six |
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
