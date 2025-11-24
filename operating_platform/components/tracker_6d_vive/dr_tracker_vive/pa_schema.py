import pyarrow as pa

pa_vec3 = pa.list_(pa.float32(), 3)  # x, y, z
pa_quaternion = pa.list_(pa.float32(), 4)  # w, x, y, z

# 定义 imu_schema
pa_imu_fields = [
    pa.field("serial_number", pa.string()),
    pa.field("acc", pa_vec3),
    pa.field("gyro", pa_vec3),
    pa.field("mag", pa_vec3),
]
pa_imu_schema = pa.schema(pa_imu_fields)

# 定义pose_schema
pa_pose_fileds = [
    pa.field("serial_number", pa.string()),
    pa.field("position", pa_vec3),
    pa.field("rotation", pa_quaternion),
]

pa_pose_schema = pa.schema(pa_pose_fileds)
