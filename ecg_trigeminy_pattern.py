"""Research-only, beat-index-based ectopy pattern audit (not a diagnosis)."""
from __future__ import annotations
from typing import Any, Mapping

VERSION = 'MEDCALC_G26_TRIGEMINY_SEQUENCE_SHADOW_V1'
PREMATURE_KINDS = frozenset(('PVC_COMPATIBLE', 'PAC_OR_NARROW_PREMATURE_BEAT'))


def audit_trigeminy(events: list[Mapping[str, Any]] | None, beat_count: int) -> dict[str, Any]:
    result = {
        'version': VERSION,
        'trigeminy_evaluable': False,
        'trigeminy_pattern': False,
        'trigeminy_repeated_cycles': 0,
        'trigeminy_pattern_kind': None,
        'trigeminy_first_index': None,
        'diagnostic_claim_allowed': False,
        'review_only': True,
    }
    try:
        n = int(beat_count)
    except (TypeError, ValueError, OverflowError):
        return result
    if n < 9 or n > 100000 or not isinstance(events, list):
        return result
    # An index can have only one premature-beat subtype. Ambiguous events
    # and duplicate indices invalidate this audit rather than boosting it.
    kinds = {}
    for row in events:
        if not isinstance(row, Mapping):
            return result
        idx = row.get('beat_index')
        if type(idx) is not int or not 0 <= idx < n or idx in kinds:
            return result
        kinds[idx] = str(row.get('kind') or '')
    best = (0, None, None)
    for index, kind in sorted(kinds.items()):
        if kind not in PREMATURE_KINDS:
            continue
        count = 1
        while (
            (index + 3 * count) in kinds
            and kinds[index + 3 * count] == kind
            and not any(j in kinds for j in (index + 3 * (count-1) + 1, index + 3 * (count-1) + 2))
        ):
            count += 1
        if count > best[0]:
            best = (count, index, kind)
    result['trigeminy_evaluable'] = True
    if best[0] >= 3:
        result.update({
            'trigeminy_pattern': True,
            'trigeminy_repeated_cycles': int(best[0]),
            'trigeminy_first_index': int(best[1]),
            'trigeminy_pattern_kind': str(best[2]),
        })
    return result