#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from robot_descriptions.loaders.yourdfpy import load_robot_description
import pyroki as pk
import jax.numpy as jnp
import jaxlie
import jax_dataclasses as jdc
import jaxls
import jax

@jdc.jit
def solve_ik_jit(
    robot: pk.Robot,
    target_se3: jaxlie.SE3,
    target_link_idx: jax.Array,
    initial_q: jax.Array,
    ) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    joint_mask = jnp.ones(robot.joints.num_actuated_joints)

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
        pk.costs.limit_constraint(
            robot,
            joint_var,
        )
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
class MyNode(Node):
    def __init__(self):
        super().__init__('test')
        

        urdf = load_robot_description("panda_description")
        self.robot = pk.Robot.from_urdf(urdf=urdf)
        self.eef_frame = "panda_hand"
        self.timer = self.create_timer(0.1, self.tick)

        dummy_se3 = jaxlie.SE3.identity()
        dummy_idx = jnp.array(self.robot.links.names.index(self.eef_frame), dtype=jnp.int32)
        dummy_q = jnp.zeros(self.robot.joints.num_actuated_joints)
        
        # This triggers the compilation
        _ = solve_ik_jit(self.robot, dummy_se3, dummy_idx, dummy_q) 
        print("⚡ JAX IK Compiler ready. Solving at microseconds.")
    def tick(self):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    
    try:
         rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
