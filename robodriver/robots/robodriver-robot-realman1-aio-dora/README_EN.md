# robodriver-robot-realman1-aio-dora
## Get Start
Clone the repository
```
git clone https://github.com/FlagOpen/RoboDriver.git
```

Install the project to Robodriver
```
cd /path/to/your/RoboDriver
pip install uv
uv venv .venv -p 3.10
source .venv/bin/activate
uv pip install -e .[hardware]
```

Install the project to robodriver-robot-realman1-aio-dora
```
cd /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-realman1-aio-dorarobodriver-robot-realman1-aio-dora
uv venv .venv -p 3.9
uv pip install -e .
source .venv/bin/activate
```

Configure the dataflow.yml
```
cd /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-realman1-aio-dorarobodriver-robot-realman1-aio-dora
cd dora
```

Open the dataflow.yml file, then modify the VIRTUAL_ENV path, as well as DEVICE_SERIAL, ARM_IP, and ARM_PORT. 
### Parameter explanation:

- VIRTUAL_ENV: The path of the virtual environment used by the realman dora node
- DEVICE_SERIAL: Serial number of the RealSense camera
- ARM_IP: IP of the robotic arm
- ARM_PORT: PORT of the robotic arm

## Start collecting
Activate environment
```
cd /path/to/your/RoboDriver
source .venv/bin/activate
```

Start Dora
```
dora up
```

Start dataflow
```
cd /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-realman1-aio-dorarobodriver-robot-realman1-aio-dora
dora start dora/dataflow.yml --uv
```

Launch RoboDriver
```
cd /path/to/your/RoboDriver
source .venv/bin/activate
python robodriver/scripts/run.py \
  --robot.type=realman1_aio_dora 
```

## Bug Fixes
1. Rerun video replay failure:  
   Edit `RoboDriver/robodriver/core/coordinator.py`, change `visual_worker(mode="distant")` to `mode="local"`.

2. OpenCV cvShowImage error on launch (`python robodriver/scripts/run.py --robot.type=realman1_aio_dora`):  
   Comment out `cv2.imshow(key, img)` and `cv2.waitKey(1)` in `robodriver/scripts/run.py`.


## Data Information
RealMan robotic arm data is transmitted by the Dora node. Each robotic arm node sends **14-dimensional information** with the following composition:

- Joint angles: Values of joints 1-7 (7 dimensions)
- Gripper state: Gripper opening/closing degree 
- End-effector pose: End Euler angles (3 dimensions, representing rotation around X/Y/Z axes) \
**When you need to modify the data information, you can modify dataflow.yml and config.py**