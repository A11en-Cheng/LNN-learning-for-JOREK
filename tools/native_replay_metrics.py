"""Pure native-array diagnostics for continuous versus checkpoint replay.

No unit conversion, state mutation, tolerance gate, or production advancement.
Use ``relative_metrics(ref[channel], replay[channel])`` for each projection
channel (and independently for its saved deltas); axes are never inferred.
"""
from __future__ import annotations

import numpy as np

_ARRAY_SHAPES = {'x': (3,), 'st': (2,), 'V_prev': (3,), 'P_imp': (11,),
                 'weight': (), 'i_elm': (), 'i_life': (), 't_birth': ()}


def _norm(values):
    scale = np.max(np.abs(values), initial=np.longdouble(0))
    return scale * np.sqrt(np.sum((values / scale) ** 2)) if scale else np.longdouble(0)


def relative_metrics(reference, candidate):
    """Compare equal-shaped finite real arrays, preserving their native units.

    L2 relative = ||candidate-reference||2 / ||reference||2. The maximum
    normalized difference uses max(abs(reference)); it is NOT pointwise.
    Pointwise ratios include only reference != 0. For nonempty all-zero
    reference AND candidate the norm/max ratios are defined as zero by the
    replay reporting convention. Nonzero/zero remains undefined. Ratios are None
    with explicit reasons, never an epsilon denominator. Empty inputs report
    no observations. Extended precision and scaled norms avoid squaring a
    large/small native value directly. JSON numbers outside float64 range are
    None with an ``undefined`` entry; representable relative ratios remain.
    """
    ref = np.asarray(reference)
    got = np.asarray(candidate)
    if ref.shape != got.shape:
        raise ValueError('reference and candidate shapes must match')
    if any(a.dtype.kind not in 'biuf' for a in (ref, got)):
        raise ValueError('finite real numeric arrays required')
    ref = ref.astype(np.longdouble, copy=False).ravel()
    got = got.astype(np.longdouble, copy=False).ravel()
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise ValueError('finite real numeric arrays required')
    difference = np.abs(got-ref)
    nonzero = ref != 0
    pointwise = difference[nonzero] / np.abs(ref[nonzero])
    rn, dn = _norm(ref), _norm(difference)
    rm = np.max(np.abs(ref), initial=np.longdouble(0))
    dm = np.max(difference, initial=np.longdouble(0))
    undefined = {}

    def number(value, key, reason=None):
        if value is None:
            undefined[key] = reason
            return None
        if not np.isfinite(value) or abs(value) > np.finfo(float).max:
            undefined[key] = 'outside_finite_float64_json_range'
            return None
        return float(value)

    empty_reason = 'empty_comparison' if not len(ref) else 'zero_reference_norm'
    result = dict(count=int(len(ref)), nonzero_reference_count=int(nonzero.sum()),
        zero_reference_count=int((~nonzero).sum()),
        zero_reference_mismatch_count=int(np.count_nonzero(difference[~nonzero])),
        exact_match_count=int(np.count_nonzero(difference == 0)),
        reference_l2=number(rn, 'reference_l2'),
        difference_l2=number(dn, 'difference_l2'),
        reference_max_abs=number(rm, 'reference_max_abs'),
        max_abs_difference=number(dm, 'max_abs_difference') if len(ref) else
            number(None, 'max_abs_difference', 'empty_comparison'),
        relative_l2=number(dn/rn if rn else (0. if len(ref) and not dn else None),
            'relative_l2', empty_reason),
        max_abs_difference_over_reference_max_abs_nonpointwise=number(
            dm/rm if rm else (0. if len(ref) and not dm else None),
            'max_abs_difference_over_reference_max_abs_nonpointwise', empty_reason),
        pointwise_relative=dict(count=int(len(pointwise)),
            max=number(pointwise.max() if len(pointwise) else None,
                'pointwise_relative.max', 'no_nonzero_reference'),
            mean=number(pointwise.mean() if len(pointwise) else None,
                'pointwise_relative.mean', 'no_nonzero_reference'),
            rms=number(_norm(pointwise)/np.sqrt(np.longdouble(len(pointwise))) if len(pointwise) else None,
                'pointwise_relative.rms', 'no_nonzero_reference')),
        undefined=undefined)
    return result


def _cloud_arrays(cloud):
    get = cloud.__getitem__ if isinstance(cloud, dict) else lambda key: getattr(cloud, key)
    arrays = {key: np.asarray(get(key)) for key in _ARRAY_SHAPES}
    n = arrays['weight'].size
    for key, tail in _ARRAY_SHAPES.items():
        if arrays[key].shape != (n, *tail):
            raise ValueError(f'cloud {key} must have shape {(n, *tail)}')
    if any(arrays[key].dtype.kind not in 'iu' for key in ('i_elm', 'i_life')):
        raise ValueError('cloud i_elm and i_life must be integer arrays')
    group = get('group')
    if not isinstance(group, str):
        raise ValueError('cloud group must be a string identity')
    arrays['group'] = group
    arrays['alive'] = arrays['i_elm'] > 0
    return arrays


def _matched_live(ref, got):
    if ref['group'] != got['group']:
        return np.empty(0, dtype=np.int64)
    n = min(len(ref['weight']), len(got['weight']))
    return np.flatnonzero(ref['alive'][:n] & got['alive'][:n] &
                          (ref['i_life'][:n] == got['i_life'][:n]))


def _circular_delta(reference, candidate):
    period = np.longdouble(2*np.pi)
    difference = np.fmod(np.asarray(candidate, dtype=np.longdouble)-
                         np.asarray(reference, dtype=np.longdouble), period)
    return np.where(difference < -period/2, difference+period,
                    np.where(difference >= period/2, difference-period, difference))


def _numeric(ref, got, ids, *, increments=False):
    result = {}
    for key, names in (('x', ('R', 'Z', 'phi')), ('st', ('s', 't')),
                       ('V_prev', ('R', 'Z', 'phi')), ('P_imp', tuple(map(str, range(11))))):
        result[key] = {}
        for j, name in enumerate(names):
            a, b = ref[key][ids, j], got[key][ids, j]
            if key == 'x' and name == 'phi':
                if not increments:
                    a = np.remainder(np.asarray(a, dtype=np.longdouble), np.longdouble(2*np.pi))
                difference = _circular_delta(a, b)
                metric = relative_metrics(a, a+difference)
                metric['difference_convention'] = 'circular_shortest_arc_[-pi,pi)'
                metric['reference_convention'] = ('signed_shortest_arc_increment' if increments else 'modulo_2pi')
            else:
                metric = relative_metrics(a, b)
            result[key][name] = metric
    result['weight'] = relative_metrics(ref['weight'][ids], got['weight'][ids])
    return result


def _identity_difference(ref, got):
    matched = _matched_live(ref, got)
    ref_only, got_only = ref['alive'].copy(), got['alive'].copy()
    ref_only[matched] = False
    got_only[matched] = False
    def identities(arrays, mask):
        return [[arrays['group'], int(i), int(arrays['i_life'][i])] for i in np.flatnonzero(mask)]
    n = min(len(ref_only), len(got_only))
    reused = (np.flatnonzero(ref['alive'][:n] & got['alive'][:n] &
                            (ref['i_life'][:n] != got['i_life'][:n])).tolist()
              if ref['group'] == got['group'] else [])
    return dict(reference_only_live_count=int(ref_only.sum()),
        candidate_only_live_count=int(got_only.sum()),
        reference_only_live_ids=identities(ref, ref_only),
        candidate_only_live_ids=identities(got, got_only), reused_live_slots=reused)


def _exact(ref, got):
    n = min(len(ref['weight']), len(got['weight']))
    result = dict(group=dict(equal=ref['group'] == got['group'],
                              reference=ref['group'], candidate=got['group']))
    for key in ('i_life', 'i_elm', 'alive', 't_birth'):
        a, b = ref[key], got[key]
        mismatches = int(np.count_nonzero(a[:n] != b[:n]))
        result[key] = dict(equal=len(a) == len(b) and mismatches == 0,
            scope='all_slots_including_inactive', compared_slots=n, mismatch_count=mismatches,
            reference_only_slots=max(0, len(a)-len(b)), candidate_only_slots=max(0, len(b)-len(a)))
    return result


def _normalization(arrays, ids):
    weight = np.asarray(arrays['weight'][ids], dtype=np.longdouble)
    probability = np.asarray(arrays['P_imp'][ids], dtype=np.longdouble)
    if not np.isfinite(weight).all() or not np.isfinite(probability).all():
        raise ValueError('normalization summary requires finite live weights/P_imp')
    row_sum = probability.sum(axis=1)
    weighted_charge = np.sum(weight[:, None]*probability, axis=0)
    total_weight = weight.sum()
    weighted_sum = weighted_charge.sum()
    undefined = {}

    def number(value, key, reason=None):
        if value is None:
            undefined[key] = reason
            return None
        if not np.isfinite(value) or abs(value) > np.finfo(float).max:
            undefined[key] = 'outside_finite_float64_json_range'
            return None
        return float(value)

    return dict(count=int(len(ids)),
        total_weight=number(total_weight, 'total_weight'),
        negative_weight_count=int(np.count_nonzero(weight < 0)),
        zero_weight_count=int(np.count_nonzero(weight == 0)),
        negative_probability_count=int(np.count_nonzero(probability < 0)),
        probability_above_one_count=int(np.count_nonzero(probability > 1)),
        probability_row_sum_not_exactly_one_count=int(np.count_nonzero(row_sum != 1)),
        probability_row_sum_max_abs_deviation_from_one=number(
            np.max(np.abs(row_sum-1)) if len(ids) else None,
            'probability_row_sum_max_abs_deviation_from_one', 'empty_comparison'),
        weight_times_probability_by_charge={str(z): number(value, f'weight_times_probability_by_charge.{z}')
            for z, value in enumerate(weighted_charge)},
        weighted_probability_sum=number(weighted_sum, 'weighted_probability_sum'),
        weighted_normalization_signed_error=number(
            (weighted_sum-total_weight)/total_weight if total_weight else None,
            'weighted_normalization_signed_error', 'zero_total_weight'),
        absolute_weighted_row_sum_error=number(
            np.sum(np.abs(weight)*np.abs(row_sum-1))/np.sum(np.abs(weight)) if np.any(weight) else None,
            'absolute_weighted_row_sum_error', 'zero_total_absolute_weight'),
        undefined=undefined)


def compare_cloud(reference, candidate, *, before=None):
    """Compare Native MarkerCloud objects (or mappings of the same arrays).

    Identity is (group, implicit zero-based slot, i_life). Numeric fields use
    ONLY equal live identities; inactive storage, absent slots and reused
    identities are excluded. Components retain native units. P_imp requires
    all eleven Ne states. Phi state differences are shortest circular arcs.
    With before, increments compare only live identities persisting through
    all three clouds. Phi increments are shortest endpoint arcs (winding is
    not recoverable). Birth/loss lists are endpoint identity differences,
    not an event log of transient particles born and lost between endpoints.
    Exact discrete fields/t_birth compare every slot, including inactive
    storage. Normalization summarizes all live rows on each side and also
    their common identity subset; invalid finite probabilities are reported
    without clipping or renormalization. Clock objects are outside this API.
    """
    ref, got = _cloud_arrays(reference), _cloud_arrays(candidate)
    ids = _matched_live(ref, got)
    result = dict(schema='native-cloud-replay-metrics-v1', units='native_input_arrays_no_conversion',
        identity_definition='(group, zero_based_slot, i_life)',
        counts=dict(reference_slots=len(ref['weight']), candidate_slots=len(got['weight']),
            reference_live=int(ref['alive'].sum()), candidate_live=int(got['alive'].sum()),
            reference_inactive_excluded=int((~ref['alive']).sum()),
            candidate_inactive_excluded=int((~got['alive']).sum()), matched_live=int(len(ids))),
        identity=_identity_difference(ref, got), exact=_exact(ref, got),
        numeric=_numeric(ref, got, ids), increments=None,
        normalization=dict(scope='all_live_per_side_plus_matched_live_subset',
            reference=_normalization(ref, np.flatnonzero(ref['alive'])),
            candidate=_normalization(got, np.flatnonzero(got['alive'])),
            matched_live=dict(reference=_normalization(ref, ids), candidate=_normalization(got, ids))))
    if before is not None:
        initial = _cloud_arrays(before)
        persisting = np.intersect1d(ids, _matched_live(initial, ref), assume_unique=True)
        deltas = []
        for arrays in (ref, got):
            values = {key: np.asarray(arrays[key][persisting], dtype=np.longdouble)-
                           np.asarray(initial[key][persisting], dtype=np.longdouble)
                      for key in ('x', 'st', 'V_prev', 'P_imp', 'weight')}
            values['x'][:, 2] = _circular_delta(initial['x'][persisting, 2], arrays['x'][persisting, 2])
            deltas.append(values)
        result['increments'] = dict(count=int(len(persisting)),
            scope='same_identity_live_in_before_reference_and_candidate',
            numeric=_numeric(*deltas, np.arange(len(persisting)), increments=True))
        changes = {}
        for name, arrays in (('reference', ref), ('candidate', got)):
            difference = _identity_difference(initial, arrays)
            changes[name] = dict(birth_count=difference['candidate_only_live_count'],
                loss_count=difference['reference_only_live_count'],
                birth_ids=difference['candidate_only_live_ids'], loss_ids=difference['reference_only_live_ids'],
                reused_live_slots=difference['reused_live_slots'])
        result['identity']['from_before'] = changes
        result['identity']['from_before_scope'] = 'endpoint_identity_changes_not_full_event_log'
    return result
