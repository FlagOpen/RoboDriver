read -p "请输入机器人的名称，可输入的有realman、aloha、pika、so101、galbot_g1: " robot_type
python3 operating_platform/scripts/run.py \
  --robot.type=$robot_type