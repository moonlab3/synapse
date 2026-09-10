from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as MsgDuration


def build_target_messages(action_chunk: list, dt: float, targets: dict) -> dict:
    """
    action_chunk: list[dict[component_name, JointState]] -- brain adapter output.
    dt:           seconds between consecutive waypoints (adapter.chunk_dt).
    targets:      component_name -> ResolvedTopic | None, from
                  EmbodimentParser.get_component_targets(muscle_option).

    Returns component_name -> msg, ready to publish. Components with no
    resolved target are silently omitted -- never invent a fallback.
    """
    per_component_steps = {}
    for step in action_chunk:
        for component_name, js in step.items():
            per_component_steps.setdefault(component_name, []).append(js)

    out = {}
    for component_name, waypoints in per_component_steps.items():
        resolved = targets.get(component_name)
        if resolved is None:
            continue

        if resolved.msg_type == 'JointTrajectory':
            out[component_name] = _to_joint_trajectory(waypoints, dt)
        elif resolved.msg_type == 'JointState':
            out[component_name] = waypoints[-1]
        else:
            raise ValueError(f"❌ No converter for msg_type '{resolved.msg_type}' ({component_name})")

    return out


def _to_joint_trajectory(waypoints: list, dt: float) -> JointTrajectory:
    traj = JointTrajectory()
    traj.joint_names = waypoints[0].name

    for i, js in enumerate(waypoints):
        point = JointTrajectoryPoint()
        point.positions = list(js.position)
        if i > 0:
            prev = waypoints[i - 1].position
            point.velocities = [(p - c) / dt for p, c in zip(js.position, prev)]
        else:
            point.velocities = [0.0] * len(js.position)
        t = dt * (i + 1)
        point.time_from_start = MsgDuration(sec=int(t), nanosec=int((t % 1) * 1e9))
        traj.points.append(point)

    return traj