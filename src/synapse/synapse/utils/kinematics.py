"""Shared pyroki/JAX kinematics for brain adapters.

One copy of the IK solver, the FK pass and the per-robot pyroki setup that
ManualAdapter and GR00TAdapter (and any future adapter that needs Cartesian
targets) all used to carry privately. Adapters never touch jax directly:
poses cross this boundary as [x, y, z, roll, pitch, yaw] lists and joints as
plain position sequences.
"""
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as pk
import yourdfpy
from loguru import logger
from robot_descriptions.loaders.yourdfpy import load_robot_description

from synapse.utils.assets_pathfinder import assets_get_path

logger.disable("jaxls")
logger.disable("pyroki")


@jdc.jit
def solve_ik_jit(
    robot: pk.Robot,
    target_se3: jaxlie.SE3,
    target_link_idx: jnp.ndarray,
    initial_q: jnp.ndarray,
    joint_mask: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled IK solver for extreme performance."""
    joint_var = robot.joint_var_cls(0)

    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            target_se3,
            target_link_idx,
            pos_weight=1.0,
            ori_weight=1.0,
            joint_mask=joint_mask
        ),
        pk.costs.limit_constraint(robot, joint_var)
    ]

    init_vals = jaxls.VarValues.make([joint_var.with_value(initial_q)])

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            initial_vals=init_vals,
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0)
        )
    )
    return sol[joint_var]


def se3_to_list(se3: jaxlie.SE3) -> list:
    """jaxlie.SE3 -> [x, y, z, roll, pitch, yaw]."""
    t = se3.translation()
    rpy = se3.rotation().as_rpy_radians()
    return [float(t[0]), float(t[1]), float(t[2]),
            float(rpy[0]), float(rpy[1]), float(rpy[2])]


def list_to_se3(pose: list) -> jaxlie.SE3:
    """[x, y, z, roll, pitch, yaw] -> jaxlie.SE3."""
    rotation = jaxlie.SO3.from_rpy_radians(pose[3], pose[4], pose[5])
    return jaxlie.SE3.from_rotation_and_translation(rotation, jnp.array(pose[:3]))


def interpolate_pose(pose_a: list, pose_b: list, alpha: float) -> list:
    """Constant-rate screw interpolation from pose_a to pose_b.

    Interpolating the two poses as SE3 rather than lerping the raw
    [x, y, z, r, p, y] lists: RPY lerp breaks near +/-pi and bends the path.
    """
    a, b = list_to_se3(pose_a), list_to_se3(pose_b)
    return se3_to_list(a @ jaxlie.SE3.exp(alpha * (a.inverse() @ b).log()))


def pose_distance(pose_a: list, pose_b: list) -> tuple:
    """(metres, radians) between two poses -- what a move's duration is built from."""
    delta = list_to_se3(pose_a).inverse() @ list_to_se3(pose_b)
    return (float(jnp.linalg.norm(delta.translation())),
            float(jnp.linalg.norm(delta.rotation().log())))


class ArmKinematics:
    """FK and IK for every manipulator in one embodiment.

    Holds the pyroki robots, their eef frames and their DOF masks, plus
    current_eef_poses -- the same dict the scenario_parser pose checks read off
    the adapter, so an adapter just keeps a reference to it.
    """

    def __init__(self, robots_cfg: dict, terminal=None):
        self.robots_cfg = robots_cfg
        self.terminal = terminal
        self.robots = {}
        self.eef_frame = {}
        self.dof_indices = {}
        self.dof_mask = {}
        self.current_eef_poses = {}

        dummy_se3 = jaxlie.SE3.identity()

        for name, cfg in robots_cfg.items():
            if cfg.get('type') != 'manipulator':
                continue

            if cfg.get('yourdfpy_description'):
                urdf = load_robot_description(cfg.get('description_name'))
            else:
                urdf = yourdfpy.URDF.load(assets_get_path(cfg.get('urdf_filename')))

            self.robots[name] = pk.Robot.from_urdf(urdf=urdf)
            self.eef_frame[name] = cfg.get('eef_frame')
            # Deliberately not pre-seeded with zeros: the scenario pose checks read
            # this dict, and a missing entry means "no pose observed yet" while a
            # zero entry reads as "at the origin" and trips pose_bound on tick one.

            expected_dofs = self.robots[name].joints.num_actuated_joints
            self.dof_indices[name] = jnp.array(cfg.get('joint_indices'), dtype=jnp.int32)
            self.dof_mask[name] = jnp.zeros(expected_dofs).at[self.dof_indices[name]].set(1.0)

            if terminal is not None:
                terminal.debug(f"[{self.robots[name].links.names}]")
                terminal.log(f"[{name}] actuated joints: {self.robots[name].joints.actuated_names}")

            # Compile both now, so the first real waypoint is not a compile. FK is
            # jitted inside pyroki and costs seconds on its first call, which lands
            # on whichever inference runs first otherwise.
            dummy_q = jnp.zeros(expected_dofs)
            solve_ik_jit(self.robots[name], dummy_se3, self._eef_idx(name),
                         dummy_q, self.dof_mask[name])
            self.robots[name].forward_kinematics(dummy_q)

    @property
    def manipulator_names(self) -> list:
        return list(self.robots)

    def _eef_idx(self, name: str) -> jnp.ndarray:
        return jnp.array(self.robots[name].links.names.index(self.eef_frame[name]), dtype=jnp.int32)

    def seed_from_positions(self, name: str, positions) -> jnp.ndarray:
        """Component joint positions -> this robot's full actuated-DOF vector.

        Extra values are dropped and missing ones zero-filled, so a component
        whose JointState disagrees with joint_indices still solves.
        """
        idx = self.dof_indices[name]
        n = idx.shape[0]
        q = jnp.array(positions)
        if q.shape[0] > n:
            q = q[:n]
        elif q.shape[0] < n:
            q = jnp.pad(q, (0, n - q.shape[0]))
        return jnp.zeros(self.robots[name].joints.num_actuated_joints).at[idx].set(q)

    def eef_pose(self, name: str, positions) -> list:
        """Eef pose of one manipulator at the given joint positions. Pure: unlike
        forward_kinematics it does not touch current_eef_poses, so a caller can
        check where a solved waypoint actually lands."""
        link_poses = self.robots[name].forward_kinematics(self.seed_from_positions(name, positions))
        return se3_to_list(jaxlie.SE3(link_poses[self.robots[name].links.names.index(self.eef_frame[name])]))

    def forward_kinematics(self, joints_dict: dict) -> dict:
        """Eef pose per manipulator. A component with no joint state keeps its
        last known pose, so callers always get an entry for every manipulator."""
        eef_poses = {}

        for name in self.robots:
            joint_state = joints_dict.get(name)

            if not joint_state or not joint_state.position:
                eef_poses[name] = self.current_eef_poses.get(name, [0.0] * 6).copy()
                if self.terminal is not None:
                    self.terminal.debug(f"[{name}] no joint state, reusing last eef pose")
                continue

            eef_poses[name] = self.eef_pose(name, joint_state.position)
            self.current_eef_poses[name] = eef_poses[name].copy()

        return eef_poses

    def solve(self, name: str, target_pose: list, seed_q: jnp.ndarray) -> jnp.ndarray:
        """One IK solve. seed_q is a full-DOF vector -- see seed_from_positions.
        Returns the solved full-DOF vector, usable as the next seed."""
        return solve_ik_jit(self.robots[name], list_to_se3(target_pose),
                            self._eef_idx(name), seed_q, self.dof_mask[name])
