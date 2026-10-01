"""Controller variant: the original simulate_task with an optional integral term in drive-phase heading control.

Built by patching forward_model.simulate_task's own source in memory, so every
other line is identical; forward_model.py on disk is not modified. With ki=0 the
variant reproduces the original exactly (see test_controller_variants.py).
The integral is limited so ki * integral never exceeds the 30% correction clamp.
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import forward_model  # noqa: E402

_PATCHES = [
    ('def simulate_task(theta_j, theta_m, theta_b, theta_t, task_name, rng=None):',
     'def simulate_task_variant(theta_j, theta_m, theta_b, theta_t, task_name, rng=None, ki=0.0):'),
    ('        prev_heading_error = 0.0\n',
     '        prev_heading_error = 0.0\n        heading_integral = 0.0\n'),
    ('                h_correction = KP_HEADING * h_error + KD_HEADING * (h_error - prev_heading_error) / dt_s\n',
     '                if ki > 0:\n'
     '                    limit = 0.3 * command_power / ki\n'
     '                    heading_integral = min(max(heading_integral + h_error * dt_s, -limit), limit)\n'
     '                h_correction = KP_HEADING * h_error + KD_HEADING * (h_error - prev_heading_error) / dt_s + ki * heading_integral\n'),
]


def _build():
    source = inspect.getsource(forward_model.simulate_task)
    for old, new in _PATCHES:
        if source.count(old) != 1:
            raise RuntimeError(f'Patch target not found exactly once: {old.strip()[:60]}')
        source = source.replace(old, new)
    # Defined in forward_model's namespace so constant overrides on that module apply.
    exec(compile(source, '<simulate_task_variant>', 'exec'), forward_model.__dict__)
    return forward_model.__dict__['simulate_task_variant']


simulate_task_variant = _build()
