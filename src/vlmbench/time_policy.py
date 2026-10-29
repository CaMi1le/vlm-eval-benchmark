"""Declared response deadlines: preserve raw evidence and retain failed cases."""
from .common import json_hash

NONTHINKING_SECONDS = 900


def enforce_deadline(row, seconds=NONTHINKING_SECONDS, timeout=None, phase='nonthinking'):
    """A late natural EOS is still a timeout; an existing shorter timeout stays."""
    late = row.get('status') == 'ok' and (row.get('total_seconds') or 0) > seconds
    forced = timeout and (row['id'], row['condition']) == (timeout['id'], timeout['condition'])
    if not late and not forced:
        return dict(row)
    return {**row, 'status': 'timeout', 'answer': '', 'final_answer_text': '',
            'model_case_abandoned': True, 'request_deadline_seconds': seconds,
            'observed_total_seconds': row.get('total_seconds'),
            'observed_wait_seconds': timeout.get('observed_elapsed_seconds') if forced else row.get('total_seconds'),
            'total_seconds': seconds, 'discarded_raw_row_sha256': json_hash(row),
            'error': ('Nonthinking' if phase=='nonthinking' else phase)+' response exceeded the declared deadline; this model abandons this case.',
            'timeout_policy': {'schema': phase+'-900s-20261020', 'seconds': seconds,
                               'source': 'live_watchdog' if forced else 'completed_after_deadline',
                               'kept_in_denominator': True, 'retry_timed_out_case': False}}
