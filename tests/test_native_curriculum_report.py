"""Keep the readable field table tied to actual loss contributions and nodes."""
from tools.report_native_cuda_curriculum import FIELDS, rows_for


def test_rows_keep_field_order_contributions_and_forecast_node():
    events = []
    for step in (1, 2):
        events.append({
            'forecast_step': step,
            'time_si': step * 0.001,
            'loss': {'per_field_total_contribution': [step * (i + 1) for i in range(9)]},
            'metrics': {'fields': {'gauss_values': {
                name: {'relative_l2': step / (i + 1), 'pearson_r': 0.9}
                for i, name in enumerate(FIELDS)
            }}},
        })
    rows = rows_for({'window': 'fixture:1', 'events': events}, 'training', 3)
    assert len(rows) == 18
    assert tuple(row['field'] for row in rows[:9]) == FIELDS
    assert sum(row['loss_contribution'] for row in rows[:9]) == 45
    assert rows[-1]['relative_l2'] == 2 / 9
    assert rows[-1]['forecast_step'] == 2
    assert rows[-1]['epoch'] == 3
    assert rows[-1]['time_si'] == 0.002


def test_rows_do_not_replace_undefined_relative_error_or_correlation():
    event = {
        'forecast_step': 1, 'time_si': 0.001,
        'loss': {'per_field_total_contribution': [0.] * 9},
        'metrics': {'fields': {'gauss_values': {
            name: {'relative_l2': None, 'pearson_r': None} for name in FIELDS
        }}},
    }
    rows = rows_for({'window': 'constant', 'events': [event]}, 'final')
    assert all(row['relative_l2'] is None and row['pearson_r'] is None for row in rows)
