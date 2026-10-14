"""Bounded CPU lookup approximation experiment on a real first source window.

One birth realization and identical post-birth RNG are shared by native and
lookup candidates. Fixed initial native fields/raw history are inputs; there
is no neural forecast, checkpoint restoration, supervision or optimization.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import resource
import signal
import time
import traceback

import numpy as np


def _rng():
    import torch
    return random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()


def _restore(state):
    import torch
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def _rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _json(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray):
        if value.size > 4096:
            raise ValueError('Large array must be compacted before JSON serialization')
        return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def _compact_diagnostics(value):
    """Keep scalar controls; summarize arrays without retaining large lists."""
    from tools.diagnose_native_real_chain import _array_digest
    if isinstance(value, dict):
        return {key: _compact_diagnostics(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_compact_diagnostics(item) for item in value]
    if not isinstance(value, np.ndarray): return value
    a = np.asarray(value)
    summary = dict(shape=list(a.shape), dtype=str(a.dtype), hash=_array_digest(a))
    if a.dtype.kind not in 'biuf': return summary
    finite = np.isfinite(a)
    summary['nonfinite_count'] = int(a.size-np.count_nonzero(finite))
    if summary['nonfinite_count']:
        summary.update(l2=None, total=None, undefined='nonfinite_array')
        return summary
    flat = a.ravel()
    scale = np.longdouble(np.max(np.abs(flat), initial=0))
    square_sum = np.longdouble(0); total = np.longdouble(0)
    for start in range(0, len(flat), 65536):
        block = flat[start:start+65536].astype(np.longdouble)
        total += block.sum()
        if scale: square_sum += np.sum((block/scale)**2)
    def number(x):
        return float(x) if np.isfinite(x) and abs(x) <= np.finfo(float).max else None
    summary.update(l2=number(scale*np.sqrt(square_sum)), total=number(total))
    if a.ndim == 2:
        summary['total_by_channel'] = [number(x) for x in np.sum(a, axis=0, dtype=np.longdouble)]
    return summary


def _sampling_probe(lookup, before, births, fields_current, chunk_size):
    """Diagnostic-only private sampling, bound to this advance's actual tables.

Native points remains the full public oracle. No field/feedback snapshots are
created here, and all initial live plus all live born samples are retained.
"""
    from tools.diagnose_native_real_chain import _array_digest
    from tools.native_replay_metrics import relative_metrics
    fields, feedback = lookup.field_owner, lookup.feedback_owner
    if fields is None or feedback is None:
        raise ValueError('Sampling probe requires tables populated by source.advance')
    if (not np.array_equal(fields, fields_current) or
            not np.array_equal(feedback, before.feedback.values)):
        raise ValueError('Actual source table owners differ from fixed initial fields/feedback')
    table_refs = lookup.field_table, lookup.feedback_table
    groups = (before.cloud, births)
    slots = [np.flatnonzero(cloud.alive) for cloud in groups]
    elements = np.concatenate([cloud.i_elm[ids].astype(np.int64)-1
                               for cloud, ids in zip(groups, slots)])
    local = np.concatenate([cloud.st[ids] for cloud, ids in zip(groups, slots)])
    expected_values = []; actual_values = []; expected_gradients = []; actual_gradients = []
    expected_feedback = []; actual_feedback = []
    for start in range(0, len(elements), chunk_size):
        e, st = elements[start:start+chunk_size], local[start:start+chunk_size]
        native_values, native_gradients, _, _, _ = lookup.native.points(e, st, fields)
        native_feedback, _, _, _, _ = lookup.native.points(e, st, feedback)
        values, gradients, projected, _ = lookup._marker_points(e, st, fields, feedback)
        expected_values.append(native_values); actual_values.append(values)
        expected_gradients.append(native_gradients); actual_gradients.append(gradients)
        expected_feedback.append(native_feedback); actual_feedback.append(projected)
    if (lookup.field_owner is not fields or lookup.feedback_owner is not feedback or
            lookup.field_table is not table_refs[0] or lookup.feedback_table is not table_refs[1]):
        raise ValueError('Sampling probe rebuilt or rebound actual event tables')
    result = dict(scope='diagnostic_only_initial_live_and_shared_births_not_trajectory',
        initial_live_count=len(slots[0]), live_birth_count=len(slots[1]), count=len(elements), cropped=False,
        fields_owner_digest=_array_digest(fields), feedback_owner_digest=_array_digest(feedback),
        element_local_digest=_array_digest((elements, local)), owners_and_tables_preserved=True,
        field_names=['psi', 'Phi', 'Z_j', 'omega', 'rho', 'T_i', 'v_par', 'rho_imp', 'T_e'],
        fields={}, feedback={})
    if not len(elements): return result
    v, vg, g, gg, p, pg = [np.concatenate(parts) for parts in
        (expected_values, actual_values, expected_gradients, actual_gradients,
         expected_feedback, actual_feedback)]
    for j, name in enumerate(result['field_names']):
        result['fields'][name] = dict(values=relative_metrics(v[:, j], vg[:, j]),
            gradients={direction: relative_metrics(g[:, j, k], gg[:, j, k])
                       for k, direction in enumerate(('R', 'Z'))})
    for j, name in enumerate(('dE_ion', 'rad_sink', 'Z_eff', 'Z_imp', 'N_imp')):
        result['feedback'][name] = relative_metrics(p[:, j], pg[:, j])
    return result


def _components(marker):
    from tools.diagnose_native_real_chain import _array_digest
    return dict(cloud={k: _array_digest(getattr(marker.cloud, k)) for k in marker.cloud.array_names()},
        transport_history=_array_digest(getattr(marker,'transport_history',None)),
        feedback={k: _array_digest(getattr(marker.feedback, k)) for k in ('values', 'deltas')},
        clocks=_array_digest((marker.clock, marker.feedback.time_si, marker.feedback.gather_time_si)),
        metadata=_array_digest((marker.provenance, marker.feedback.layout, marker.feedback.history_provenance)))


def _compare(reference, candidate, before):
    from tools.native_replay_metrics import compare_cloud, relative_metrics
    from tools.diagnose_native_real_chain import _array_digest
    full_slots = {}
    for name in reference.cloud.array_names():
        a, b = getattr(reference.cloud, name), getattr(candidate.cloud, name)
        entry = dict(shape=list(a.shape), array_equal=bool(np.array_equal(a, b)))
        if a.shape != b.shape:
            entry['candidate_shape'] = list(b.shape)
        elif name in ('i_elm', 'i_life', 't_birth'):
            entry['mismatches'] = int(np.count_nonzero(a != b))
        else:
            entry['relative_metrics'] = relative_metrics(a, b)
        full_slots[name] = entry
    channels = ('dE_ion', 'rad_sink', 'Z_eff', 'Z_imp', 'N_imp')
    feedback = {kind: {name: dict(relative_metrics=relative_metrics(
        getattr(reference.feedback, kind)[j], getattr(candidate.feedback, kind)[j]),
        array_equal=bool(np.array_equal(getattr(reference.feedback, kind)[j],
                                       getattr(candidate.feedback, kind)[j])))
        for j, name in enumerate(channels)} for kind in ('values', 'deltas')}
    return dict(cloud=compare_cloud(reference.cloud, candidate.cloud, before=before.cloud),
        cloud_full_slots=full_slots, feedback=feedback,
        clocks_exact=_components(reference)['clocks'] == _components(candidate)['clocks'],
        clock_reference=vars(reference.clock), clock_candidate=vars(candidate.clock),
        feedback_time_reference_si=reference.feedback.time_si,
        feedback_time_candidate_si=candidate.feedback.time_si,
        metadata_exact=_array_digest((reference.provenance, reference.feedback.layout)) ==
                       _array_digest((candidate.provenance, candidate.feedback.layout)))


def _postadvance_outliers(native, lookup, before, reference, candidate, chunk_size):
    """Endpoint evidence only; no inferred substep path or causal explanation."""
    from tools.native_replay_metrics import relative_metrics
    from tools.diagnose_native_real_chain import _array_digest
    initial, ref, got = before.cloud, reference.cloud, candidate.cloud
    if not (initial.group == ref.group == got.group):
        raise ValueError('Outlier report requires matching cloud group identities')
    if not (len(initial.weight) == len(ref.weight) == len(got.weight)):
        raise ValueError('Outlier report requires full matching slot inventories')
    matched = np.flatnonzero(ref.alive & got.alive & (ref.i_life == got.i_life))
    persistent = matched[initial.alive[matched] & (initial.i_life[matched] == ref.i_life[matched])]
    def arc(a, b):
        period = np.longdouble(2*np.pi)
        d = np.fmod(np.asarray(b, dtype=np.longdouble)-np.asarray(a, dtype=np.longdouble), period)
        return np.where(d < -period/2, d+period, np.where(d >= period/2, d-period, d))
    ref_inc = ref.x[persistent].astype(np.longdouble)-initial.x[persistent]
    got_inc = got.x[persistent].astype(np.longdouble)-initial.x[persistent]
    ref_inc[:, 2] = arc(initial.x[persistent, 2], ref.x[persistent, 2])
    got_inc[:, 2] = arc(initial.x[persistent, 2], got.x[persistent, 2])
    inc_diff = got_inc-ref_inc
    inc_diff[:, 2] = arc(ref_inc[:, 2], got_inc[:, 2])
    velocity_diff = got.V_prev[persistent].astype(np.longdouble)-ref.V_prev[persistent]
    report = dict(scope='postadvance_endpoints_no_substep_trace',
        matched_live_count=len(matched), initially_alive_same_identity_count=len(persistent),
        increment_convention='R_Z_native_m; phi_shortest_signed_endpoint_arc_[-pi,pi)',
        velocity_convention='V_prev_R_Z_phi_components_m_per_s; direct_component_difference',
        top_selection='descending_absolute_component_difference_stable_slot_order',
        top_record_inventory='top10 slot_id joins slots keyed by decimal slot id',
        squared_error_scope='entire_initially_alive_same_identity_matched_set_per_component',
        zero_total_squared_error_ratio_convention=0.,
        increment_metrics={}, velocity_metrics={}, top10={}, cumulative_top10_ratio={},
        slots={}, geometry_consistency={}, success=False)
    selected = set()
    for j, direction in enumerate(('R', 'Z', 'phi')):
        report['increment_metrics'][direction] = relative_metrics(ref_inc[:, j], ref_inc[:, j]+inc_diff[:, j])
        report['velocity_metrics'][direction] = relative_metrics(ref.V_prev[persistent, j], got.V_prev[persistent, j])
        for kind, difference in (('increment', inc_diff), ('V_prev', velocity_diff)):
            order = np.argsort(-np.abs(difference[:, j]), kind='stable')[:10]
            square = difference[:, j]**2
            total_square = np.sum(square, dtype=np.longdouble)
            rows = [dict(slot_id=int(persistent[k]), signed_difference=float(difference[k, j]),
                         absolute_difference=float(abs(difference[k, j])),
                         squared_error_fraction=float(square[k]/total_square) if total_square else 0.)
                    for k in order]
            key = kind+'_'+direction
            report['top10'][key] = rows
            report['cumulative_top10_ratio'][key] = (
                float(np.sum(square[order], dtype=np.longdouble)/total_square) if total_square else 0.)
            report.setdefault('total_squared_error', {})[key] = float(total_square)
            selected.update(row['slot_id'] for row in rows)
    def probability_summary(probability):
        return dict(sum=float(np.sum(probability)), min=float(np.min(probability)),
            max=float(np.max(probability)), mean_charge=float(np.dot(np.arange(len(probability)), probability)),
            dominant_charge=int(np.argmax(probability)))
    for slot in sorted(selected):
        k = int(np.searchsorted(persistent, slot))
        record = dict(slot_id=slot, group=ref.group, lifetime=int(ref.i_life[slot]),
            reference_increment_RZphi=ref_inc[k].astype(float).tolist(),
            candidate_increment_RZphi=got_inc[k].astype(float).tolist(),
            increment_difference_RZphi=inc_diff[k].astype(float).tolist(),
            velocity_difference_RZphi=velocity_diff[k].astype(float).tolist())
        for name, cloud in (('before', initial), ('reference', ref), ('candidate', got)):
            record[name] = dict(element_id=int(cloud.i_elm[slot]), st=cloud.st[slot].tolist(),
                RZphi=cloud.x[slot].tolist(), V_prev=cloud.V_prev[slot].tolist(),
                weight=float(cloud.weight[slot]), P_imp=probability_summary(cloud.P_imp[slot]))
        report['slots'][str(slot)] = record
    # Every matched live endpoint, including shared births, independently
    # evaluated with native cubic geometry, not lookup forward geometry.
    for name, cloud in (('reference', ref), ('candidate', got)):
        residual = np.empty((len(matched), 2))
        for start in range(0, len(matched), chunk_size):
            ids = matched[start:start+chunk_size]
            try:
                _, _, rz, _, _ = native.points(cloud.i_elm[ids].astype(np.int64)-1,
                                                cloud.st[ids], native.coordinates[:1])
                if not np.isfinite(rz).all():
                    raise ValueError('Nonfinite native endpoint geometry observation')
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                report['geometry_consistency'][name] = dict(success=False,
                    all_matched_finite_verified=False, completed_count=start, required_count=len(matched),
                    failed_batch_slot_ids=ids.tolist(),
                    id_scope='failed_batch_not_individually_confirmed_invalid_slots',
                    error_type=type(error).__name__, error=str(error))
                report['failed_probe'] = 'native_endpoint_geometry_'+name
                return report
            residual[start:start+len(ids)] = rz-cloud.x[ids, :2]
        norm = np.linalg.norm(residual, axis=1)
        report['geometry_consistency'][name] = dict(success=True, all_matched_finite_verified=True,
            count=len(matched), units='m',
            residual_convention='native_cubic_forward_RZ_minus_cloud_RZ',
            max=float(norm.max()) if len(norm) else None,
            p99=float(np.quantile(norm, .99)) if len(norm) else None,
            RMS=float(np.sqrt(np.mean(norm**2))) if len(norm) else None,
            components={direction: dict(max_abs=float(np.max(np.abs(residual[:, j]))) if len(norm) else None,
                p99_abs=float(np.quantile(np.abs(residual[:, j]), .99)) if len(norm) else None,
                RMS=float(np.sqrt(np.mean(residual[:, j]**2))) if len(norm) else None)
                for j, direction in enumerate(('R', 'Z'))})
    # Native field evaluation only at the bounded union of top slots, using
    # the actual event owner (fixed U), never a recorded future endpoint.
    fields = lookup.field_owner
    if fields is None: raise ValueError('Endpoint field diagnostic requires actual source field owner')
    report['endpoint_native_field_owner_digest'] = _array_digest(fields)
    report['endpoint_field_semantics'] = 'native_cubic_at_reported_endpoint_e_st_fixed_actual_source_U_not_substep_samples'
    ids = np.asarray(sorted(selected), dtype=np.int64)
    for name, cloud in (('reference', ref), ('candidate', got)):
        for start in range(0, len(ids), chunk_size):
            batch = ids[start:start+chunk_size]
            try:
                values, gradients, _, _, _ = native.points(cloud.i_elm[batch].astype(np.int64)-1,
                                                           cloud.st[batch], fields)
                if not np.isfinite(values).all() or not np.isfinite(gradients).all():
                    raise ValueError('Nonfinite endpoint native field/gradient observation')
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                report.update(failed_probe='native_endpoint_fields_'+name,
                    failed_batch_slot_ids=batch.tolist(), error_type=type(error).__name__, error=str(error))
                return report
            for k, slot in enumerate(batch):
                report['slots'][str(slot)][name]['native_endpoint_fields'] = dict(
                    values={field: float(values[k, j]) for j, field in enumerate(
                        ('psi', 'Phi', 'Z_j', 'omega', 'rho', 'T_i', 'v_par', 'rho_imp', 'T_e'))},
                    grad_rho_imp_RZ=gradients[k, 7].tolist(), grad_Phi_RZ=gradients[k, 1].tolist())
    report['success'] = True
    return report


def _original_einsum_tabulate(lookup, fields):
    """Frozen original full table refresh, including allocation/finite guard."""
    n = len(fields)
    q = lookup.grid_points**2
    out = np.empty((lookup.n_elements, q, n), dtype=np.float64)
    width = max(1, lookup.build_chunk//q)
    for start in range(0, lookup.n_elements, width):
        end = min(start+width, lookup.n_elements)
        ids = lookup.native.vertex[:, start:end].T
        coefficients = fields[:, :, 0, :][:, :, ids].transpose(2, 0, 3, 1)
        coefficients = coefficients*lookup.native.size[:, :, start:end].transpose(2, 1, 0)[:, None]
        out[start:end] = np.einsum('qvd,efvd->eqf', lookup.basis, coefficients)
    out = out.reshape(lookup.n_elements, lookup.grid_points, lookup.grid_points, n)
    if not np.isfinite(out).all(): raise ValueError('Nonfinite original einsum table')
    out.flags.writeable = False
    return out


def _refresh_table_metrics(reference, candidate):
    """Bounded-memory global norms; no denominator epsilon or array copies."""
    result = dict(reference_shape=list(reference.shape), candidate_shape=list(candidate.shape),
                  reference_dtype=str(reference.dtype), candidate_dtype=str(candidate.dtype))
    if reference.shape != candidate.shape:
        return dict(result, success=False, reason='shape_mismatch')
    # flat slicing bounds scratch allocations even for a non-C current table.
    a, b = reference.flat, candidate.flat
    reference_scale = np.longdouble(0); difference_scale = np.longdouble(0)
    mismatch_count = 0
    for start in range(0, len(a), 65536):
        x = a[start:start+65536].astype(np.longdouble)
        y = b[start:start+65536].astype(np.longdouble)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return dict(result, success=False, reason='nonfinite_table', failed_flat_offset=start)
        reference_scale = max(reference_scale, np.max(np.abs(x), initial=np.longdouble(0)))
        difference_scale = max(difference_scale, np.max(np.abs(y-x), initial=np.longdouble(0)))
        mismatch_count += int(np.count_nonzero(x != y))
    reference_squares = np.longdouble(0); difference_squares = np.longdouble(0)
    for start in range(0, len(a), 65536):
        x = a[start:start+65536].astype(np.longdouble)
        difference = b[start:start+65536].astype(np.longdouble)-x
        if reference_scale: reference_squares += np.sum((x/reference_scale)**2)
        if difference_scale: difference_squares += np.sum((difference/difference_scale)**2)
    rn = reference_scale*np.sqrt(reference_squares)
    dn = difference_scale*np.sqrt(difference_squares)
    relative = dn/rn if rn else (np.longdouble(0) if len(a) and not dn else None)
    def number(x):
        return float(x) if x is not None and np.isfinite(x) and abs(x) <= np.finfo(float).max else None
    return dict(result, success=True, count=len(a), array_equal=mismatch_count == 0,
        mismatch_count=mismatch_count, max_abs_difference=number(difference_scale),
        reference_l2=number(rn), difference_l2=number(dn), relative_l2=number(relative),
        relative_l2_undefined=None if relative is not None else 'zero_reference_norm_or_empty',
        norm_convention='global_native_units_scaled_longdouble_reduction_no_epsilon')


def _refresh_diagnostic(lookup, repeats, memory_mib):
    """Independent snapshot misses, never frozen-owner cache hits or source time."""
    from tools.diagnose_native_real_chain import _array_digest
    if repeats < 3: raise ValueError('Independent refresh diagnostic requires at least three repeats')
    owner = lookup.field_owner
    if owner is None or owner.flags.writeable:
        raise ValueError('Actual source read-only field owner required')
    owner_digest = _array_digest(owner)
    fixed_refs = lookup.field_owner, lookup.feedback_owner, lookup.field_table, lookup.feedback_table
    # Keep every input strongly owned until the diagnostic ends; no identity
    # recycling can mimic reuse. Each algorithm sees exactly the same snapshot.
    inputs = []
    for ordinal in range(repeats+1):
        fields = np.array(owner, dtype=owner.dtype, copy=True)
        fields *= 1.+(ordinal+1)*2.**-40
        fields.flags.writeable = False
        inputs.append(fields)
    input_digests = [_array_digest(fields) for fields in inputs]
    if len(set(input_digests+[owner_digest])) != repeats+2:
        raise ValueError('Refresh perturbations failed to create distinct actual numerical inputs')
    report = dict(success=False, scope='bilinear_field_table_refresh_only_after_complete_source',
        actual_field_owner_digest=owner_digest, repeats=repeats,
        input_rule='new_owned_readonly_copy_times_1_plus_(ordinal_plus_1)*2**-40; identical_input_for_all_algorithms',
        input_digests=input_digests, process_peak_rss_before_bytes=_rss(),
        lookup_table_bytes_before=lookup.table_bytes, source_stats_before=deepcopy(lookup.stats),
        cold_warmup={}, iterations=[], timings_exclude='input_preparation_hashes_metrics_gc_warmup_and_source_advance',
        cold_scope='first_independent_refresh_with_new_snapshot_in_already_built_process_not_process_cold',
        warm_scope='post_warmup_refreshes_each_with_distinct_snapshot_not_same_identity_cache_hits',
        original_algorithm='frozen_einsum_qvd_efvd_to_eqf_in_original_element_chunks')
    tick = time.perf_counter()
    buffer = np.empty((lookup.n_elements, lookup.grid_points, lookup.grid_points, len(owner)), dtype=np.float64)
    allocation_seconds = time.perf_counter()-tick
    address = int(buffer.__array_interface__['data'][0])
    if any(np.shares_memory(buffer, table) for table in fixed_refs[2:] if table is not None):
        raise ValueError('Diagnostic refresh buffer aliases actual source table')
    report['dedicated_reuse_buffer'] = dict(allocation_seconds=allocation_seconds,
        bytes=buffer.nbytes, byte_address=address, aliases_actual_source_tables=False,
        allocation_excluded_from_refresh_timings=True)
    operations = dict(original_einsum=lambda fields: _original_einsum_tabulate(lookup, fields),
                      current_tabulate=lookup._tabulate,
                      current_reused_buffer=lambda fields: lookup._tabulate(fields, out=buffer))
    def run(name, fields):
        if _rss() > memory_mib*1024**2: raise MemoryError('Refresh diagnostic RSS budget exceeded')
        tick = time.perf_counter()
        table = operations[name](fields)
        seconds = time.perf_counter()-tick
        if _rss() > memory_mib*1024**2: raise MemoryError('Refresh diagnostic RSS budget exceeded')
        record = dict(seconds=seconds, output_bytes=table.nbytes,
            peak_rss_bytes=_rss(), lookup_table_bytes=lookup.table_bytes,
            shape=list(table.shape), dtype=str(table.dtype),
            returned_byte_address=int(table.__array_interface__['data'][0]))
        if name == 'current_reused_buffer':
            record.update(returned_same_object=table is buffer,
                          byte_address_reused=record['returned_byte_address'] == address)
            if not record['returned_same_object'] or not record['byte_address_reused']:
                raise ValueError('Current refresh failed to reuse diagnostic buffer')
        return table, record
    # One first-snapshot warmup per algorithm, excluded from repeated timings.
    report['cold_warmup_order'] = list(operations)
    for name in operations:
        table, record = run(name, inputs[0])
        report['cold_warmup'][name] = dict(record, excluded_from_warm_timings=True)
        del table
    for ordinal in range(repeats):
        names = list(operations)
        shift = ordinal % len(names)
        order = names[shift:]+names[:shift]
        tables = {}; iteration = dict(ordinal=ordinal, input_digest=input_digests[ordinal+1],
                                      order=order, algorithms={})
        for name in order:
            tables[name], iteration['algorithms'][name] = run(name, inputs[ordinal+1])
        tick = time.perf_counter()
        iteration['metrics'] = _refresh_table_metrics(tables['original_einsum'], tables['current_tabulate'])
        iteration['reused_buffer_metrics'] = _refresh_table_metrics(tables['original_einsum'], tables['current_reused_buffer'])
        iteration['allocated_vs_reused_metrics'] = _refresh_table_metrics(tables['current_tabulate'], tables['current_reused_buffer'])
        iteration['comparison_seconds'] = time.perf_counter()-tick
        report['iterations'].append(iteration)
        del tables; gc.collect()
    if (lookup.field_owner is not fixed_refs[0] or lookup.feedback_owner is not fixed_refs[1]
            or lookup.field_table is not fixed_refs[2] or lookup.feedback_table is not fixed_refs[3]
            or _array_digest(owner) != owner_digest):
        raise ValueError('Independent refresh diagnostic changed actual source owners/tables')
    report.update(success=all(row[key]['success'] for row in report['iterations']
        for key in ('metrics', 'reused_buffer_metrics', 'allocated_vs_reused_metrics')),
        owners_and_tables_preserved=True, process_peak_rss_after_bytes=_rss(),
        lookup_table_bytes_after=lookup.table_bytes, source_stats_after=deepcopy(lookup.stats),
        warm_seconds={name: [row['algorithms'][name]['seconds'] for row in report['iterations']]
                      for name in operations})
    return report


def execute(args, result):
    import torch
    from tools.diagnose_native_real_chain import Stages, _initial_pairing, _array_digest
    from tools.native_real_chain_builder import build_real
    from src.models.native_coupling import source_time_sampler
    from src.models import native_lookup_experiment
    from src.models.native_lookup_experiment import ExperimentalLookupMesh, make_lookup_source

    stages = Stages()
    start = time.monotonic()
    def timed(name, operation):
        result['stage'] = name
        if _rss() > args.memory_mib*1024**2: raise MemoryError('RSS budget exceeded')
        with stages.track(name):
            tick = time.perf_counter()
            try:
                value = operation()
                if _rss() > args.memory_mib*1024**2: raise MemoryError('RSS budget exceeded')
                return value
            finally:
                result['timings_seconds'][name] = time.perf_counter()-tick
                result['peak_rss_bytes'] = _rss()

    config = json.loads(args.config.read_text())
    specs = [w for w in config['windows'] if w['split'] == 'train']
    if len(specs) != 2: raise ValueError('Exactly two declared phaseA train anchors required')
    spec = specs[args.run_index]; run = Path(spec['run']).resolve(); anchor = (4110, 5680)[args.run_index]
    expected = 'phaseA_dataset_1d22_ang15_massloss_neon_v400' + ('_injInner' if args.run_index else '')
    if spec['anchor'] != anchor or run.name != expected: raise ValueError('Run/index/anchor mismatch')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    result.update(run=str(run), anchor=anchor, torch_threads=torch.get_num_threads(),
                  numpy_version=np.__version__, torch_version=torch.__version__,
                  lookup_module=dict(path=native_lookup_experiment.__file__, sha256=hashlib.sha256(
                      Path(native_lookup_experiment.__file__).read_bytes()).hexdigest()))
    controls = timed('initial_pairing', lambda: _initial_pairing(run, anchor, stages))
    model, source, shard, runtime, provider = timed('autonomous_build', lambda:
        build_real(run, anchor, controls, stages, args, feedback_mode='autonomous'))
    before = runtime.marker
    if len(before.cloud.weight) != 1000000: raise ValueError('Full real 1M-slot cloud required')
    initial_digest = _components(before)
    immutable_digest = _array_digest((runtime.fields_previous, runtime.fields_current,
                                     runtime.fluid_deltas, runtime.shards))
    target = runtime.fluid_time_si
    def sampler():
        return source_time_sampler(source, before, runtime.fields_current,
            fluid_deltas=runtime.fluid_deltas, completed_dt_si=runtime.fluid_dt_previous_si)
    birth_sampler = sampler()
    births = timed('shared_births', lambda: shard.births(runtime.shards, before, target,
        runtime.fields_previous, runtime.fields_current, source, time_sampler=birth_sampler))
    post_birth_rng = _rng()
    birth_digest = _array_digest(births)
    result.update(initial_components=initial_digest, initial_fields_history_shards_digest=immutable_digest,
        post_birth_rng_digest=_array_digest(post_birth_rng), births_digest=birth_digest,
        births_count=len(births.weight), target_si=target, initial_cloud_slots=len(before.cloud.weight),
        initial_live_count=int(before.cloud.alive.sum()), source_configuration=source.configuration,
        source_fingerprint=_array_digest(source.configuration), model_fingerprint=_array_digest(model.checkpoint_configuration),
        initial_record=runtime.marker.provenance, controls=controls,
        source_window=dict(start_si=before.clock.sim_time_si, target_si=target,
            field_start_si=before.clock.field_start_si, field_end_si=before.clock.field_end_si,
            completed_fluid_dt_si=runtime.fluid_dt_previous_si, static=source.field_static),
        scope='first_complete_source_window_fixed_initial_U_rawD_feedback_not_neural_chain')
    # The unused model/equilibrium are deliberately not executed after builder.
    del model, provider
    def advance(adapter, key):
        # Input clone/sampler/RNG restoration are identical, outside source timer.
        candidate = deepcopy(before); shared_births = deepcopy(births); sample = sampler()
        _restore(post_birth_rng)
        out, diagnostics = timed(key, lambda: adapter.advance(candidate, target,
            runtime.fields_previous, runtime.fields_current, births=shared_births, time_sampler=sample))
        evidence = dict(components=_components(out), rng_digest=_array_digest(_rng()),
            live_count=int(out.cloud.alive.sum()), slots=len(out.cloud.weight),
            diagnostics=_compact_diagnostics(diagnostics))
        if _array_digest(births) != birth_digest or _components(before) != initial_digest:
            raise ValueError('Shared birth/initial marker input mutated')
        if _array_digest((runtime.fields_previous, runtime.fields_current, runtime.fluid_deltas,
                          runtime.shards)) != immutable_digest:
            raise ValueError('Fixed initial fields/history/shards mutated')
        return out, evidence
    baseline, evidence = advance(source, 'native_source_advance')
    result['baseline'] = evidence
    for grid in args.grid_points:
        entry = dict(grid_points=grid, approximation=True, success=False)
        result['candidates'].append(entry)
        lookup = timed(f'g{grid}_geometry_build', lambda: ExperimentalLookupMesh(source.mesh,
            grid_points=grid, build_chunk=args.build_chunk, max_bytes=args.lookup_max_bytes,
            sampling=args.sampling))
        entry['sampling'] = args.sampling
        entry['lookup_configuration'] = deepcopy(lookup.configuration)
        entry['lookup_fingerprint'] = _array_digest(lookup.configuration)
        entry.update(geometry_build_seconds=lookup.geometry_build_seconds,
                     geometry_table_bytes=lookup.table_bytes, estimated_bytes=lookup.estimated_bytes)
        # Factory is inside the timer: any eager field-table preparation counts.
        candidate = deepcopy(before); shared_births = deepcopy(births); sample = sampler()
        _restore(post_birth_rng)
        holder = {}
        def operation():
            adapter = make_lookup_source(source, lookup); holder['adapter'] = adapter
            return adapter.advance(candidate, target, runtime.fields_previous, runtime.fields_current,
                                   births=shared_births, time_sampler=sample)
        out, diagnostics = timed(f'g{grid}_source_factory_and_advance', operation)
        entry.update(source_configuration=deepcopy(holder['adapter'].configuration),
            source_success=True,
            source_fingerprint=_array_digest(holder['adapter'].configuration),
            components=_components(out), rng_digest=_array_digest(_rng()),
            diagnostics=_compact_diagnostics(diagnostics),
            slots=len(out.cloud.weight), live_count=int(out.cloud.alive.sum()),
            source_stats_before_probe=deepcopy(lookup.stats), final_table_bytes=lookup.table_bytes,
            peak_rss_bytes=_rss())
        del diagnostics
        entry['sampling_probe'] = timed(f'g{grid}_diagnostic_sampling_probe', lambda:
            _sampling_probe(lookup, before, births, runtime.fields_current, args.chunk_size))
        entry['source_stats_after_probe'] = deepcopy(lookup.stats)
        entry['probe_field_table_builds_unchanged'] = (
            entry['source_stats_before_probe']['field_table_builds'] ==
            entry['source_stats_after_probe']['field_table_builds'])
        if not entry['probe_field_table_builds_unchanged']:
            raise ValueError('Diagnostic sampling probe rebuilt field tables')
        entry['probe_rng_unchanged'] = entry['rng_digest'] == _array_digest(_rng())
        if not entry['probe_rng_unchanged']: raise ValueError('Diagnostic sampling probe consumed RNG')
        entry['metrics'] = timed(f'g{grid}_comparison', lambda: _compare(baseline, out, before))
        entry['postadvance_outliers'] = timed(f'g{grid}_postadvance_outliers', lambda:
            _postadvance_outliers(source.mesh, lookup, before, baseline, out, args.chunk_size))
        entry['postadvance_diagnostic_rng_unchanged'] = entry['rng_digest'] == _array_digest(_rng())
        if not entry['postadvance_diagnostic_rng_unchanged']:
            raise ValueError('Postadvance diagnostic consumed RNG')
        if args.refresh_repeats:
            entry['table_refresh_diagnostic'] = timed(f'g{grid}_independent_table_refresh', lambda:
                _refresh_diagnostic(lookup, args.refresh_repeats, args.memory_mib))
            entry['refresh_rng_unchanged'] = entry['rng_digest'] == _array_digest(_rng())
            if not entry['refresh_rng_unchanged']:
                raise ValueError('Independent table refresh consumed RNG')
        entry['rng_exact'] = evidence['rng_digest'] == entry['rng_digest']
        entry['same_initial_inputs'] = (_components(before) == initial_digest and
            _array_digest(births) == birth_digest and _array_digest((runtime.fields_previous,
            runtime.fields_current, runtime.fluid_deltas, runtime.shards)) == immutable_digest)
        if not entry['same_initial_inputs']: raise ValueError('Candidate mutated shared bound inputs')
        entry['diagnostics_success'] = (entry['postadvance_outliers']['success'] and
            entry.get('table_refresh_diagnostic', {}).get('success', True))
        entry['success'] = entry['source_success'] and entry['diagnostics_success']
        del out, candidate, holder, lookup, shared_births, sample; gc.collect()
    result.update(success=all(entry['success'] for entry in result['candidates']),
                  stage='completed', total_seconds=time.monotonic()-start,
                  final_peak_rss_bytes=_rss())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-index', type=int, choices=(0, 1), required=True)
    p.add_argument('--config', type=Path, default=Path('config/native-autonomous-training-pilot.json'))
    p.add_argument('--output', type=Path, required=True, help='Exclusive result JSON')
    p.add_argument('--grid-points', type=int, nargs='+', default=[17, 33, 65])
    p.add_argument('--sampling', choices=('bilinear', 'native_cubic_confirmed'), default='bilinear')
    p.add_argument('--refresh-repeats', type=int, default=0,
                   help='Bilinear-only independent old/new field-table refresh timings; 0 disables, otherwise >=3')
    p.add_argument('--chunk-size', type=int, default=2048)
    p.add_argument('--memory-mib', type=int, default=8192)
    p.add_argument('--wall-seconds', type=int, default=600)
    p.add_argument('--birth-max-proposals', type=int, default=1000000)
    p.add_argument('--build-chunk', type=int, default=32768)
    p.add_argument('--lookup-max-bytes', type=int, default=4*1024**3)
    p.add_argument('--seed', type=int, default=20261009)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args(); args.spatial_backend = 'batched_v1'
    if not 0 < args.wall_seconds <= 900 or min(args.chunk_size, args.memory_mib,
            args.birth_max_proposals, args.build_chunk, args.lookup_max_bytes, args.threads) <= 0:
        p.error('Positive resource limits and wall-seconds <=900 required')
    if min(args.grid_points) < 3 or len(set(args.grid_points)) != len(args.grid_points):
        p.error('Unique grid-points >=3 required')
    if args.refresh_repeats != 0 and args.refresh_repeats < 3:
        p.error('--refresh-repeats must be 0 or >=3')
    if args.refresh_repeats and args.sampling != 'bilinear':
        p.error('--refresh-repeats applies only to bilinear sampling')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(success=False, stage='start', settings=vars(args), timings_seconds={}, candidates=[],
        tool=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        training=False, optimizer_updates=0, device='cpu', future_field_feedback=False,
        sampling=args.sampling,
        birth_timing='one shared realization excluded from lookup speed comparison',
        source_timing='native advance versus lookup factory+advance including cold event field tables; geometry separate',
        metrics_timing='separate from source wall timing',
        sampling_probe_timing='separate diagnostic call after source.advance; actual field/feedback owners reused',
        execution_order='native_then_declared_grids')
    def timeout(*_): raise TimeoutError('Lookup experiment total wall budget exceeded')
    with args.output.open('x') as stream:
        start = time.monotonic()
        previous = signal.signal(signal.SIGALRM, timeout); signal.alarm(args.wall_seconds)
        try: execute(args, result)
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                          final_peak_rss_bytes=_rss())
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous)
            result['process_wall_seconds'] = time.monotonic()-start
            json.dump(result, stream, default=_json, indent=2, allow_nan=False)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
