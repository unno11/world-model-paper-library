"""Small numerical examples for geometry, feedback, dynamics and planning."""

import argparse
import itertools
import json

import numpy as np


def transforms():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [1., 2., 0.]
    point = np.array([1., 0., 0., 1.])
    direction = np.array([1., 0., 0., 0.])
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    np.testing.assert_allclose(transform @ point, [1., 3., 0., 1.])
    np.testing.assert_allclose(transform @ direction, [0., 1., 0., 0.])
    np.testing.assert_allclose(inverse @ transform, np.eye(4))
    q = np.array([0., 0., 0., 1.])  # xyzw
    angle = 2 * np.arccos(np.clip(abs(q @ (-q)), 0., 1.))
    assert angle == 0.
    return {'transformed_point': (transform @ point)[:3].tolist(),
            'transformed_direction': (transform @ direction)[:3].tolist(),
            'equivalent_quaternion_angle_rad': float(angle)}


def fk(q):
    q1, q2 = q
    return np.array([np.cos(q1) + .5 * np.cos(q1 + q2),
                     np.sin(q1) + .5 * np.sin(q1 + q2)])


def jacobian(q):
    q1, q2 = q
    return np.array([[-np.sin(q1) - .5 * np.sin(q1 + q2), -.5 * np.sin(q1 + q2)],
                     [np.cos(q1) + .5 * np.cos(q1 + q2), .5 * np.cos(q1 + q2)]])


def kinematics():
    endpoint = fk([np.pi / 2, -np.pi / 2])
    np.testing.assert_allclose(endpoint, [.5, 1.])
    q = np.array([.4, -.7])
    eps = 1e-6
    numeric = np.column_stack([(fk(q + eps * e) - fk(q - eps * e)) / (2 * eps)
                               for e in np.eye(2)])
    error = float(np.max(np.abs(numeric - jacobian(q))))
    assert error < 1e-8
    singular_rank = int(np.linalg.matrix_rank(jacobian([0., 0.])))
    assert singular_rank == 1
    # Local damped IK, without collision or joint-limit constraints.
    q = np.array([.3, -.8])
    target = np.array([.8, .6])
    for _ in range(100):
        j = jacobian(q)
        q += .5 * j.T @ np.linalg.solve(j @ j.T + 1e-6 * np.eye(2), target - fk(q))
    ik_error = float(np.linalg.norm(target - fk(q)))
    assert ik_error < 1e-7
    return {'fk_endpoint_m': endpoint.tolist(), 'jacobian_max_error': error,
            'straight_arm_jacobian_rank': singular_rank, 'local_ik_error_m': ik_error}


def feedback():
    # Euler-discretized integrator: x_next = x + dt * k * (goal - x).
    def run(dt, gain):
        x = 0.
        for _ in range(10):
            x += dt * gain * (1. - x)
        return x
    stable = run(.1, 2.)
    unstable = run(.1, 25.)
    np.testing.assert_allclose(stable, 1. - .8 ** 10)
    assert abs(1. - unstable) > 1.
    return {'stable_position_at_step_10': stable,
            'unstable_error_magnitude_at_step_10': abs(1. - unstable),
            'scope': 'First-order velocity-input system; no delay or saturation'}


def world_model():
    rng = np.random.default_rng(42)
    training = rng.uniform(-1., 1., size=(100, 2))
    target = .9 * training[:, 0] + .2 * training[:, 1]
    weights = np.linalg.lstsq(training, target, rcond=None)[0]
    np.testing.assert_allclose(weights, [.9, .2], atol=1e-12)
    actual = estimated = ignored_action = .7
    errors, ignored_errors = [], []
    # New state/action trajectory, not the regression training pairs.
    for action in np.linspace(-.5, .5, 20):
        actual = .9 * actual + .2 * action
        estimated = weights[0] * estimated + weights[1] * action
        ignored_action = .9 * ignored_action
        errors.append(abs(actual - estimated))
        ignored_errors.append(abs(actual - ignored_action))
    assert max(errors) < 1e-12
    assert max(ignored_errors) > .01
    return {'learned_state_action_coefficients': weights.tolist(),
            'held_out_rollout_max_error': max(errors),
            'ignoring_actions_rollout_max_error': max(ignored_errors),
            'scope': 'Noiseless scalar linear dynamics, not a neural or visual model'}


def mpc():
    # All 5^4 candidates; exact minimization only over this finite grid.
    candidates = np.array(list(itertools.product(np.linspace(-.25, .25, 5), repeat=4)))
    def plan(x):
        states = x + candidates.cumsum(axis=1)
        costs = ((states[:, -1] - 1.) ** 2
                 + .1 * np.square(states[:, :-1] - 1.).sum(axis=1)
                 + .1 * np.square(candidates).sum(axis=1))
        return candidates[np.argmin(costs)]
    open_plan = plan(0.)
    open_x = closed_x = 0.
    for step in range(10):
        open_x += open_plan[step] if step < len(open_plan) else 0.
        closed_x += plan(closed_x)[0]
        if step == 1:  # Same external disturbance, after executing second action.
            open_x -= .1
            closed_x -= .1
    assert np.max(np.abs(candidates)) <= .25
    assert abs(1. - closed_x) < abs(1. - open_x)
    return {'candidates_per_plan': len(candidates), 'open_loop_final_position': float(open_x),
            'mpc_final_position': float(closed_x), 'action_limit': .25,
            'scope': 'Exact scalar model; finite action grid; no obstacle constraints'}


EXAMPLES = {'transforms': transforms, 'kinematics': kinematics, 'feedback': feedback,
            'world_model': world_model, 'mpc': mpc}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', choices=['all', *EXAMPLES], nargs='?', default='all')
    selected = parser.parse_args().example
    names = EXAMPLES if selected == 'all' else [selected]
    print(json.dumps({name: EXAMPLES[name]() for name in names}, indent=2, allow_nan=False))
