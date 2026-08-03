"""Per-module capacity estimation for a series-connected battery pack.

Because all 9 modules in a pack share one current/Ah axis (series string), a
pack-level discharge test stops the instant the single globally-weakest cell
crosses the low-voltage cutoff -- every other (healthier) module's own true
capacity is right-censored in that same test: we only observe part of its
curve, not the point where IT would cross cutoff.

This module estimates each module's own capacity (the Ah at which its own
weakest cell would cross that test's cutoff voltage) via a tiered method,
most to least trustworthy:

  Tier 0 - measured:              module's own min-cell trace actually
                                   crosses cutoff within observed data.
  Tier 1 - extrapolated_self:     module hasn't crossed, but has already
                                   entered a real terminal knee within
                                   observed data -- extrapolate its own
                                   accelerating post-knee decay.
  Tier 2 - extrapolated_shape_transfer: module shows no knee onset yet --
                                   borrow the SAME test's real crossing
                                   module as a shape template for the
                                   terminal decay, anchored onto this
                                   module's own local trend.

A naive linear tail-slope extrapolation is also implemented, as a named
baseline that Tier 1/2 must be validated against (see
validate_module_extrapolation.py) -- it badly overestimates capacity because
real discharge curves accelerate into a knee rather than staying linear.
"""

import numpy as np

from curve_utils import detect_knee_index


def find_crossing_index(ah, v, cutoff_v, tol=0.05):
    """First index of the contiguous run, ending at the last sample, where v
    stays at/below cutoff_v (+tol for sensor noise) -- i.e. a debounced
    "sustained crossing", not a transient sag-and-recover dip. Returns None
    if the trace never reaches cutoff within observed data.

    tol was 0.01 until a real case surfaced it as too tight: cutoff_v comes
    from the raw file's own last-row pack-wide min_v, while `v` here is the
    RESAMPLED per-module trace (extract_and_resample_curve) -- when a
    cleaned file's discharge window shifts by even 1-3 rows (e.g. different
    cleaning runs of the identical raw source landing on slightly different
    start/end boundaries), the resampled trace's last point can end up just
    outside the old 0.01V tolerance even for the genuinely-correct weakest
    module, silently dropping that module (and via build_module_dataset.py,
    that whole file's pack+C-rate) out of training entirely. Measured across
    4 real affected files: the correct module missed by 0.011-0.030V every
    time, while the next-closest module was always 0.17-0.73V further away
    -- 0.05 comfortably covers every observed miss with a wide margin still
    well short of any real risk of picking the wrong module."""
    v = np.asarray(v)
    if v[-1] > cutoff_v + tol:
        return None
    idx = len(v) - 1
    while idx > 0 and v[idx - 1] <= cutoff_v + tol:
        idx -= 1
    return idx


def _interp_crossing_ah(ah, v, idx, cutoff_v):
    if idx == 0:
        return float(ah[0])
    v0, v1 = v[idx - 1], v[idx]
    ah0, ah1 = ah[idx - 1], ah[idx]
    if v0 == v1:
        return float(ah1)
    frac = (v0 - cutoff_v) / (v0 - v1)
    frac = min(max(frac, 0.0), 1.0)
    return float(ah0 + frac * (ah1 - ah0))


def _plateau_voltage(ah, v, mid_window=(0.2, 0.6)):
    """A module's own stable mid-discharge voltage level -- the baseline
    against which its terminal sag is measured, since modules differ in
    absolute voltage offset even when equally healthy."""
    total = ah[-1] - ah[0]
    lo, hi = mid_window
    mask = (ah - ah[0] > lo * total) & (ah - ah[0] < hi * total)
    if mask.sum() >= 3:
        return float(np.mean(v[mask]))
    return float(np.mean(v[:max(3, len(v) // 4)]))


def detect_gated_knee(ah, v, tail_frac=0.15, slope_ratio_threshold=2.0, mid_window=(0.2, 0.6)):
    """detect_knee_index's chord-distance method is prone to false positives
    at a search-window edge when there's no real bend. Gate it: only accept
    a knee that (a) falls in the last tail_frac of the OBSERVED span, and
    (b) is followed by a segment whose average slope is markedly steeper
    than the mid-discharge (plateau-region) segment slope -- both slopes
    fit over whole segments, not point-wise derivatives, since point-wise
    second derivatives of real (noisy) voltage traces are too noisy to gate
    on reliably (verified: rejected a real, visually-obvious knee on actual
    pack data because mid-plateau point curvature was nearly as large as the
    knee's own point curvature, purely from sample-to-sample noise).
    Returns the accepted knee index, or None if no real knee onset yet."""
    idx = detect_knee_index(ah, v, search_start_pct=0.5, search_end_pct=1.0, fallback=None)
    if idx is None:
        return None
    total = ah[-1] - ah[0]
    if total <= 0:
        return None
    if (ah[idx] - ah[0]) < (1 - tail_frac) * total:
        return None
    if idx < 3 or (len(ah) - 1 - idx) < 3:
        return None

    lo, hi = mid_window
    mid_mask = (ah - ah[0] > lo * total) & (ah - ah[0] < hi * total)
    if mid_mask.sum() >= 3:
        slope_before, _ = np.polyfit(ah[mid_mask], v[mid_mask], 1)
    else:
        slope_before, _ = np.polyfit(ah[:idx + 1], v[:idx + 1], 1)
    slope_after, _ = np.polyfit(ah[idx:], v[idx:], 1)

    if slope_after >= -1e-6:
        return None
    if abs(slope_after) < slope_ratio_threshold * abs(slope_before) + 1e-9:
        return None
    return idx


# Real-data-measured systematic bias in the exponential Tier-1 fit below:
# re-truncating each of the ~10 real Tier-0 (measured) crossings at 98%/99.5%
# of its true capacity and running this same fit on the truncated curve
# UNDERESTIMATES the true (later-measured) capacity every single time (20/20
# truncations, never once an overestimate). The bias scales with how far
# PAST THE LAST ACTUALLY-OBSERVED ROW the fit is extrapolating -- not with
# the knee-to-crossing span (which includes territory already covered by
# real data) -- so it's applied only to the unobserved portion below.
# LOPO-validated (excluding each curve when fitting its own correction):
# raw MAE 0.63 Ah -> 0.28 Ah. See validate_module_extrapolation.py for the
# re-truncation methodology.
TIER1_BIAS_CORRECTION_K = 0.31


def _extrapolate_self_exponential(ah, v, knee_idx, cutoff_v, max_extra_ah_frac):
    """Fit the module's own observed post-knee segment as an accelerating
    decay (ln(v_knee - v) linear in Ah-since-knee) and extrapolate to cutoff_v.
    Returns (capacity_ah, ok, diagnostics)."""
    ah_k, v_k = ah[knee_idx], v[knee_idx]
    seg_ah, seg_v = ah[knee_idx:], v[knee_idx:]
    if len(seg_ah) < 5:
        return None, False, {'reason': 'too_few_post_knee_points'}

    eps = 1e-4
    d = np.clip(v_k - seg_v, eps, None)
    x = seg_ah - ah_k
    c1, c0 = np.polyfit(x, np.log(d), 1)
    if c1 <= 0:
        return None, False, {'reason': 'non_accelerating_fit', 'fit_c1': float(c1)}

    target_d = v_k - cutoff_v
    if target_d <= eps:
        return float(ah_k), True, {'reason': 'knee_at_cutoff'}

    extra_x = (np.log(target_d) - c0) / c1
    if extra_x < 0:
        extra_x = 0.0
    total_span = ah[-1] - ah[0]
    max_extra = max_extra_ah_frac * total_span
    capped = False
    if extra_x > max_extra:
        extra_x, capped = max_extra, True

    raw_capacity = ah_k + extra_x
    unobserved_extra = max(0.0, raw_capacity - ah[-1])
    corrected_capacity = raw_capacity + TIER1_BIAS_CORRECTION_K * unobserved_extra

    return float(corrected_capacity), True, {'capped': capped, 'fit_c0': float(c0), 'fit_c1': float(c1)}


def _shape_transfer(ah, v, template_ah, template_v, tail_fit_frac, min_pre_knee_slope, max_extra_ah_frac):
    """Borrow the template (Tier-0) module's real post-knee shape.

    Works in DELTA-FROM-OWN-PLATEAU voltage space, not raw voltage: modules
    differ in absolute voltage baseline even when equally healthy, so "reached
    the template's knee-onset voltage" is not a comparable trigger across
    modules (verified on real data: it made several genuinely different
    healthy modules collapse to one identical predicted capacity, because
    they'd all already dropped numerically below the template's absolute
    knee voltage while still clearly in their own gentle pre-knee decline).
    Instead: express the template's knee-onset as how far it had sagged below
    ITS OWN mid-discharge plateau, fit the target's own local linear trend in
    that same relative-to-its-own-plateau space, and extrapolate to find
    where the target reaches an equal relative sag. This keeps modules
    differentiated by their own current depth-of-sag and slope."""
    diag = {}
    t_idx = detect_gated_knee(template_ah, template_v)
    if t_idx is None:
        t_idx = int(len(template_ah) * 0.9)
        diag['template_knee_fallback'] = True
    template_plateau = _plateau_voltage(template_ah, template_v)
    ah_k_t = template_ah[t_idx]
    delta_k_t = template_v[t_idx] - template_plateau  # negative: sag below its own plateau
    post_knee_span = float(template_ah[-1] - ah_k_t)
    # The template's own post-knee decay, as (Ah since its knee onset) ->
    # (sag below its own plateau) -- used below both to extrapolate INTO the
    # knee and, when the target is already past that point, to look up how
    # far along this same curve the target's current sag corresponds to.
    template_post_x = template_ah[t_idx:] - ah_k_t
    template_post_delta = template_v[t_idx:] - template_plateau

    target_plateau = _plateau_voltage(ah, v)
    n = len(ah)
    tail_n = max(5, int(n * tail_fit_frac))
    seg_ah, seg_v = ah[-tail_n:], v[-tail_n:]
    slope, intercept = np.polyfit(seg_ah, seg_v, 1)
    if slope > min_pre_knee_slope:
        slope = min_pre_knee_slope
        diag['slope_floored'] = True

    ah_last, v_last = ah[-1], v[-1]
    sag_now = v_last - target_plateau

    # Both sag_now and delta_k_t are negative (a voltage BELOW plateau) -- a
    # DEEPER sag is a MORE NEGATIVE number, so "already sagged past the
    # template's knee-onset depth" is sag_now < delta_k_t, not <=. (Bug found
    # on real data: this was inverted, so the "already past" branch below
    # never fired when it should have -- every module whose current sag
    # exceeded the template's knee depth still went through the pre-knee
    # branch, which added the template's FULL post-knee span onto ah_last
    # regardless of how much further along the equivalent curve it already
    # was, collapsing pk4 1.0C modules 2/6/8/9 to one identical prediction.)
    if sag_now >= delta_k_t:
        # Target hasn't sagged as deep (relative to its own plateau) as the
        # template had by its own knee onset yet -- extrapolate the target's
        # local pre-knee trend forward to that same relative depth, then
        # append the template's full post-knee-to-cutoff span.
        v_target_at_knee = target_plateau + delta_k_t
        ah_k_target = ah_last + (v_target_at_knee - v_last) / slope
        total_span = ah[-1] - ah[0]
        max_extra = max_extra_ah_frac * total_span
        extra_to_knee = ah_k_target - ah_last
        capped = False
        if extra_to_knee < 0:
            extra_to_knee = 0.0
        if extra_to_knee > max_extra:
            extra_to_knee, capped = max_extra, True
        capacity_ah = float(ah_last + extra_to_knee + post_knee_span)
        diag.update({'branch': 'pre_knee_extrapolation', 'capped': capped})
    else:
        # Target has ALREADY sagged (relative to its own plateau) more than
        # the template's own knee-onset depth -- it is further along the
        # equivalent post-knee curve than "just entering the knee". Locate
        # where on the template's OWN post-knee decay this same sag depth
        # occurs, and take the template's own remaining Ah from there to
        # cutoff. (Previously this branch anchored every such module at its
        # current point and always added the template's FULL post-knee span,
        # regardless of how much further past the knee onset it already was
        # -- collapsing every module that hit this branch to one identical
        # predicted capacity. Verified on real data: pk4 1.0C modules
        # 2/6/8/9 all produced the exact same value under the old logic.)
        # template_post_delta decreases monotonically with template_post_x;
        # reverse both so np.interp sees an increasing xp.
        x_equiv = float(np.interp(sag_now, template_post_delta[::-1], template_post_x[::-1]))
        remaining = max(0.0, post_knee_span - x_equiv)
        capacity_ah = float(ah_last + remaining)
        diag.update({'branch': 'matched_on_template_curve', 'x_equiv': x_equiv, 'sag_now': float(sag_now)})

    diag.update({
        'template_knee_ah': float(ah_k_t), 'template_plateau': template_plateau,
        'target_plateau': target_plateau, 'delta_k_t': float(delta_k_t),
        'post_knee_span': post_knee_span,
    })
    return capacity_ah, diag


def estimate_module_capacity(ah, module_v, cutoff_v, template_ah=None, template_v=None,
                              debounce_tol=0.01, tail_fit_frac=0.1,
                              min_pre_knee_slope=-0.005, max_extra_ah_frac=0.35):
    """Estimate one module's capacity (Ah at which its own weakest cell would
    cross cutoff_v). `module_v`/`template_v` are min-cell-voltage traces
    (see curve_utils.extract_and_resample_curve's module_min_v output), both
    sharing the same Ah axis convention as `ah`/`template_ah` respectively.
    `template_ah`/`template_v` (that test's real Tier-0 crossing module) are
    only required if this module needs Tier 2.

    Returns (capacity_ah, label_source, diagnostics).
    """
    diagnostics = {}

    idx = find_crossing_index(ah, module_v, cutoff_v, tol=debounce_tol)
    if idx is not None:
        capacity_ah = _interp_crossing_ah(ah, module_v, idx, cutoff_v)
        diagnostics['tier'] = 0
        return capacity_ah, 'measured', diagnostics

    # Physical floor for anything below: module_v was directly observed to
    # still be ABOVE cutoff_v at ah[-1] (find_crossing_index already returned
    # None), so the true crossing point cannot be earlier than ah[-1] -- any
    # extrapolation predicting less than that contradicts data we actually
    # measured. Found on real data: the least-squares exponential fit in
    # Tier 1 doesn't pass exactly through the tail's last point, and for
    # pk1 1.0C module 5 this alone pulled the predicted capacity below the
    # pack's own last observed row (147.155 Ah predicted vs. 147.194 Ah
    # actually observed with the module still above cutoff) -- impossible,
    # and it was silently accepted before this floor was added.
    observed_floor = float(ah[-1])

    knee_idx = detect_gated_knee(ah, module_v)
    if knee_idx is not None:
        capacity_ah, ok, diag = _extrapolate_self_exponential(ah, module_v, knee_idx, cutoff_v, max_extra_ah_frac)
        if ok:
            diagnostics.update(diag)
            diagnostics['tier'] = 1
            diagnostics['floor_bound'] = capacity_ah < observed_floor
            return max(capacity_ah, observed_floor), 'extrapolated_self', diagnostics
        diagnostics['tier1_rejected'] = diag

    if template_ah is None or template_v is None:
        raise ValueError(
            "Module shows no knee onset (Tier 1 not applicable) and no template "
            "curve was provided for Tier 2 shape-transfer."
        )
    capacity_ah, diag2 = _shape_transfer(ah, module_v, template_ah, template_v,
                                         tail_fit_frac, min_pre_knee_slope, max_extra_ah_frac)
    diagnostics.update(diag2)
    diagnostics['tier'] = 2
    diagnostics['floor_bound'] = capacity_ah < observed_floor
    return max(capacity_ah, observed_floor), 'extrapolated_shape_transfer', diagnostics


def estimate_module_capacity_at_targets(ah, module_v, target_cutoffs, template_ah=None, template_v=None,
                                         n_tail_points=60, **kwargs):
    """Same tiered method as estimate_module_capacity, evaluated at several
    target cutoff voltages (e.g. [3.2, 2.5]) plus a dense extrapolated tail
    curve for plotting -- built by re-evaluating that SAME fit at intermediate
    cutoff voltages, so the plotted tail and the reported target points are
    exactly consistent with each other (same underlying fit, just sampled more
    finely), rather than two independently-computed things that could disagree.

    Returns (targets, tail_ah, tail_v):
      targets  -- {cutoff_v: (capacity_ah, label_source, diagnostics)}
      tail_ah, tail_v -- arrays for the extrapolated segment beyond the last
                         observed point, down to min(target_cutoffs). Empty
                         arrays if the trace already covers that whole range.
    """
    targets = {
        cv: estimate_module_capacity(ah, module_v, cv, template_ah=template_ah, template_v=template_v, **kwargs)
        for cv in target_cutoffs
    }

    v_last = float(module_v[-1])
    v_floor = min(target_cutoffs)
    if v_last <= v_floor:
        return targets, np.array([]), np.array([])

    sample_cutoffs = np.linspace(v_last - 1e-4, v_floor, n_tail_points)
    tail_ah, tail_v = [], []
    for cv in sample_cutoffs:
        cap_ah, _source, _diag = estimate_module_capacity(
            ah, module_v, float(cv), template_ah=template_ah, template_v=template_v, **kwargs
        )
        tail_ah.append(cap_ah)
        tail_v.append(float(cv))
    return targets, np.array(tail_ah), np.array(tail_v)


def naive_linear_capacity(ah, v, cutoff_v, tail_frac=0.1):
    """Baseline for comparison only (see module docstring) -- fits a straight
    line to the last tail_frac of observed data and extrapolates it to
    cutoff_v, ignoring the real accelerating knee. Returns None if the fitted
    slope is flat/rising (can't reach a lower cutoff by extrapolation)."""
    n = len(ah)
    tail_n = max(5, int(n * tail_frac))
    seg_ah, seg_v = ah[-tail_n:], v[-tail_n:]
    slope, intercept = np.polyfit(seg_ah, seg_v, 1)
    if slope >= -1e-6:
        return None
    return float((cutoff_v - intercept) / slope)
