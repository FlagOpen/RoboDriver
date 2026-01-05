robodriver-run \
  --robot.type=agilex_aloha_aio_dora \
  --sim.arm_config='{"right": {"path": "descriptions/mjcf/piper/piper_description.xml", "pos": [0, -1, 0]}, "left": {"path": "descriptions/mjcf/piper/piper_description.xml", "pos": [0, 1, 0]}}' \
  --sim.backend=gpu \
  --sim.show_viewer=true
