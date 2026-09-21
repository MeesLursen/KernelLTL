"""Experiment 3: does the trained decoder read the embedding norm as a variance signal?

Thesis Sec. 5.2.4 argues, from the single-key cross-attention, that the decoder
is *told* the embedding norm (the Kobayashi bound) and that the norm carries the
target's variance times its registration by the anchors. Nothing in
Experiments 1-2 tests whether the decoder *uses* it. Two tests, both post-hoc:

  Test 1 (observational)   ``test1.py``   Signed variance error of the greedy
      generation, V_gen - V_target, against the norm residual u of Experiment 2,
      within variance bins, among misses. A norm-reading decoder overestimates
      the variance of targets registered louder than their variance-matched
      peers (positive within-bin slope everywhere); a decoder that merely falls
      back to its prior when the signal is faint shows a slope whose sign flips
      at the prior's own variance V_0.

  Test 2 (interventional)  ``test2.py``   Rescale each target's embedding by c
      at fixed direction (scripts/validation_variance_rescaling.py) and read
      the generated variance, validity, equivalence, relational direction and
      output entropy as functions of c. Norm-as-variance predicts generated
      variance monotone in c with direction preserved.

Layout mirrors analysis_exp1 / analysis_exp2: ``records.py`` loads and checks,
``gen_variance.py`` evaluates generated formulae on the kernel's trace sample,
``test1.py`` / ``test2.py`` compute the tables with percentile-bootstrap CIs
over targets, ``plot_exp3.py`` draws, ``run_exp3.py`` drives. Bootstrap and
JSONL loading are borrowed from analysis_exp1; the covariate derivation (u,
variance bins) is borrowed from analysis_exp2 so that u is the same u.
"""
