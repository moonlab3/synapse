#!/usr/bin/env python3
"""IK latency benchmark: GPU vs CPU JAX for ArmKinematics.solve.

Not part of the package. Run once per device with the workspace sourced:

    python3 tools/bench_ik.py --embodiment DUAL_XARM_V6 --device gpu
    python3 tools/bench_ik.py --embodiment DUAL_XARM_V6 --device cpu

Each row times one solve plus the host copy the adapters do afterwards
(to_component_positions), i.e. what one waypoint actually costs.

kinematics.py now defaults JAX to CPU (it measured ~11x faster); --device
overrides that here by setting JAX_PLATFORMS first.
"""
import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embodiment', required=True)
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--iters', type=int, default=200)
    args = parser.parse_args()

    # Must be set before jax is imported anywhere.
    os.environ['JAX_PLATFORMS'] = 'cuda' if args.device == 'gpu' else 'cpu'

    import jax
    import numpy as np
    from synapse.utils.embodiment_parser import EmbodimentParser
    from synapse.utils.kinematics import ArmKinematics

    print(f"jax devices: {jax.devices()}")
    kin = ArmKinematics(EmbodimentParser(args.embodiment).get_robots())  # compiles here

    for name in kin.manipulator_names:
        n = len(kin.dof_indices_np[name])
        positions = [0.1] * n
        target = kin.eef_pose(name, positions)
        target[2] += 0.02  # a small, reachable step like one chunk waypoint
        seed = kin.seed_from_positions(name, positions)

        for _ in range(10):  # warm-up
            kin.to_component_positions(name, kin.solve(name, target, seed))

        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            kin.to_component_positions(name, kin.solve(name, target, seed))
            samples.append(time.perf_counter() - t0)

        ms = np.array(samples) * 1000
        print(f"[{args.device}] {name}: mean {ms.mean():.2f} ms, "
              f"p50 {np.percentile(ms, 50):.2f}, p99 {np.percentile(ms, 99):.2f}, max {ms.max():.2f}")


if __name__ == '__main__':
    main()
