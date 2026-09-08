#!/bin/bash
gnome-terminal -- bash -c "
source ~/ws/xarm/install/setup.sh;
ros2 launch xarm_moveit_config dual_xarm7_moveit_realmove.launch.py robot_ip_1:=192.168.1.205 robot_ip_2:=192.168.1.221;
"

