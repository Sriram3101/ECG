# agents/validation/triage.py

def decide(flags: dict, nan_ratio: float) -> tuple:
    if flags['wrong_shape'] or flags.get('bad_dtype'):
        return 'REJECT', 'structural format failure'
    if flags['nan']:
        return ('REJECT', 'excess NaN') if nan_ratio >= 0.05 else ('REPAIR', 'sparse NaN, interpolated')
    if flags['flatline']:
        return 'REJECT', 'flatline signal'
    if flags['clipping'] or flags['abnormal_amplitude']:
        return 'REPAIR', 'saturation/outlier repaired'
    return 'PASS', 'clean signal'
