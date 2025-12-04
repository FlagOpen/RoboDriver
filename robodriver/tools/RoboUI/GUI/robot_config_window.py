#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import io, sys
import subprocess
import sys
import json
import yaml
import uuid
# import cv2
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget,
    QFrame, QFormLayout, QSpinBox, QDoubleSpinBox, QTextEdit,
    QLineEdit, QCheckBox, QComboBox, QMessageBox, QDialog, QDialogButtonBox, 
    QVBoxLayout, QScrollArea
)

from yaml_to_json import generate_config
import detect_components
from camera_widget import CameraWidget
from episode_generator import parse_robot_config_to_episode_components, generate_episode_structure
import generate_dora, generate_ros1, generate_ros2_aio
import episode_generator
import ast
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_config_json_files():
    """
    获取 ../config/ 目录下所有 JSON 文件的名称（不含路径，含后缀）
    返回：JSON 文件名列表（如 ["robot_config.json", "demo_robot.json"]）
    """
    # 拼接 config 目录路径（兼容不同操作系统）
    config_dir = os.path.join(os.path.pardir, "config")
    
    # 检查目录是否存在，不存在返回空列表
    if not os.path.exists(config_dir) or not os.path.isdir(config_dir):
        print(f"[警告] 目录 {config_dir} 不存在")
        return []
    
    # 遍历目录，筛选 .json 后缀的文件，返回文件名列表
    json_files = [
        filename for filename in os.listdir(config_dir)
        if filename.endswith(".json") and os.path.isfile(os.path.join(config_dir, filename))
    ]
    
    return json_files

# 可选：获取「不含后缀」的文件名列表（如 ["robot_config", "demo_robot"]）
def get_config_json_names_without_suffix():
    json_files = get_config_json_files()
    # 去除 .json 后缀
    return [os.path.splitext(filename)[0] for filename in json_files]


class ROS2CustomMsgConfigDialog(QDialog):
    """ROS2 自定义消息配置弹窗：仅保留 msg_type 和 msg_data_floors（复用 YAML 已有参数）"""
    """ROS2 自定义消息配置弹窗：仅保留 msg_type 和 msg_data_floors（复用 YAML 已有参数）"""
    def __init__(self, existing_config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自定义消息配置")
        self.setModal(True)
        self.resize(600, 400)
        self.existing_config = existing_config or {}  # 接收已有配置（用于加载时回显）
        # 新增：存储父窗口传递的开发模式（dora/ros1/ros2）
        self.develop_mode = parent.canvas.component_manager.mode if parent and hasattr(parent.canvas, "component_manager") else "dora"
        # 新增：动态存储 ROS 消息工具类
        self.ros_msg_tool = None
        self._init_ros_msg_tool()
    # def __init__(self, existing_config=None, parent=None):
    #     super().__init__(parent)
    #     self.setWindowTitle("自定义消息配置")
    #     self.setModal(True)
    #     self.resize(600, 400)
    #     self.existing_config = existing_config or {}  # 接收已有配置（用于加载时回显）

        layout = QVBoxLayout(self)

        # 1. 消息类型输入
        layout.addWidget(QLabel("请输入自定义消息类型（格式：包名/消息名，如：my_package/MyJointMsg）："))
        self.msg_type_edit = QLineEdit(self.existing_config.get("msgs", ""))
        self.msg_type_edit.setPlaceholderText("例如：my_package/UUState 或 custom_msgs/JointData")
        layout.addWidget(self.msg_type_edit)

        # 2. 消息格式显示框
        layout.addWidget(QLabel("\n检测到的消息格式："))
        self.msg_struct_edit = QTextEdit()
        self.msg_struct_edit.setReadOnly(True)
        layout.addWidget(self.msg_struct_edit)

        # 3. 检测按钮（移到数据层级输入之前）
        self.detect_btn = QPushButton("检测消息格式")
        self.detect_btn.clicked.connect(self._detect_msg_structure)
        layout.addWidget(self.detect_btn)

        # 4. 消息数据层级输入（复用 YAML 的 msgs_data_floors）
        layout.addWidget(QLabel("\n请输入消息数据层级（msgs_data_floors，如为空则填空）："))
        self.msg_data_floors_edit = QLineEdit(self.existing_config.get("msgs_data_floors", ""))
        self.msg_data_floors_edit.setPlaceholderText("例如：用于多层嵌套数据的层级标识，无则留空")
        layout.addWidget(self.msg_data_floors_edit)

        # 5. 确认/取消按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # 加载时如果已有消息类型，自动检测格式
        if self.existing_config.get("msgs") and not self.existing_config.get("msgs").startswith("sensor_msgs/"):
            self._detect_msg_structure()

    def _init_ros_msg_tool(self):
        """根据开发模式初始化对应的 ROS 消息工具类"""
        try:
            if self.develop_mode == "ros1":
                from ros1_msg_utils import ROSJointMsgTool
                self.ros_msg_tool = ROSJointMsgTool
            elif self.develop_mode == "ros2":
                from ros2_msg_utils import ROSJointMsgTool
                self.ros_msg_tool = ROSJointMsgTool
            else:
                self.ros_msg_tool = None
        except ImportError as e:
            print(f"[错误] 导入 ROS 消息工具失败：{e}")
            self.ros_msg_tool = None

    def _detect_msg_structure(self):
        """检测并显示消息结构（适配 ROS1/ROS2 模式）"""
        self.msg_type = self.msg_type_edit.text().strip()
        
        # 检查工具类是否初始化成功
        if not self.ros_msg_tool:
            mode_desc = "ROS1 Melodic" if self.develop_mode == "ros1" else "ROS2 Humble"
            self.msg_struct_edit.setText(f"❌ 未加载 {mode_desc} 消息工具，请检查环境和依赖")
            return
        
        # 调用对应模式的工具类方法（方法名统一，无需修改逻辑）
        if not self.ros_msg_tool.validate_msg_type(self.msg_type, parent=self):
            self.msg_struct_edit.setText(f"❌ 消息类型无效，请检查格式和 {self.develop_mode} 环境")
            return
        
        msg_struct = self.ros_msg_tool.parse_msg_structure(self.msg_type)
        if msg_struct:
            self.msg_struct_edit.setText(f"✅ 消息格式获取成功：\n\n{msg_struct}")
            QMessageBox.information(self, "检测成功", "已获取消息格式，请配置数据层级")
        else:
            self.msg_struct_edit.setText(f"❌ 未能获取消息格式，请确保消息包已编译并生效（{self.develop_mode} 环境）")
    

    def get_result(self):
        """返回配置结果（仅 msg_type 和 msg_data_floors）"""
        return {
            "msg_type": self.msg_type_edit.text().strip(),
            "msg_data_floors": self.msg_data_floors_edit.text().strip()
        }

class ComponentManager:
    def __init__(self, yaml_file="../config/components.yaml"):
        with open(yaml_file, "r", encoding="utf-8") as f:
            self.components = yaml.safe_load(f)
        # self.mode = "dora"
        #只支持ros2
        self.mode = "ros2"

    def set_mode(self, mode):
        if mode in self.components:
            # self.mode = mode
            #只支持ros2
            self.mode = "ros2"
        else:
            print(f"[WARN] 未找到模式 {mode}")

    def get_groups(self):
        return self.components.get(self.mode, {})

    def get_flat_components(self):
        flat = {}
        groups = self.get_groups()
        for group_name, group_data in groups.items():
            for comp_name, comp_info in group_data.items():
                flat[f"{group_name}/{comp_name}"] = comp_info
        return flat


# 中间画布上单个组件
class ComponentItemWidget(QWidget):
    def __init__(self, comp_type, on_delete, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # 创建标签
        self.label = QLabel(comp_type, self)
        layout.addWidget(self.label)

        # 创建删除按钮
        self.delete_button = QPushButton("删除", self)
        self.delete_button.setFixedSize(50, 25)
        self.delete_button.clicked.connect(on_delete)
        layout.addWidget(self.delete_button)

        # 将删除按钮设置在最右边
        layout.setAlignment(self.delete_button, Qt.AlignRight)
        self.setLayout(layout)


# 属性面板
class PropertyPanel(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMaximumWidth(450)
        self.setMinimumWidth(300)

        #初始化垂直布局
        self.main_layout = QVBoxLayout(self)
        
        #加一个标签
        self.title_label = QLabel("传感器属性", self)
        self.main_layout.addWidget(self.title_label)

        #加一个表单布局
        self.form_layout = QFormLayout()
        self.main_layout.addLayout(self.form_layout)

        #加一个标签
        self.info_label = QLabel("请点击画布中的组件查看属性", self)
        self.info_label.setWordWrap(True)#允许文本换行
        self.main_layout.addWidget(self.info_label)

        # 创建可编辑的下拉框
        self.editable_combo = QComboBox()
        # 关键：启用编辑模式（允许手动输入）
        self.editable_combo.setEditable(True)
        # 设置默认文本（类似你原来的 "robot_config"）
        self.editable_combo.setCurrentText("robot_config")
        # 添加预设下拉选项
        json_names = get_config_json_names_without_suffix()  # 获取不含后缀的名称
        self.editable_combo.addItems(json_names)
        self.main_layout.addWidget(QLabel("机器人名称（可输入/选择）："))
        self.main_layout.addWidget(self.editable_combo)

        # self.setLayout(self.main_layout)
        #创建一个复选框
        self.use_videos_checkbox = QCheckBox("use_videos")
        self.main_layout.addWidget(self.use_videos_checkbox)

        #创建一个按钮，绑定按钮点击事件
        self.load_button = QPushButton("加载配置")
        self.load_button.clicked.connect(self.load_config)
        self.main_layout.addWidget(self.load_button)

        self.save_button = QPushButton("保存配置")
        self.save_button.clicked.connect(self.save_config)
        self.main_layout.addWidget(self.save_button)

        self.apply_button = QPushButton("应用配置")
        self.apply_button.clicked.connect(self.apply_config)
        self.main_layout.addWidget(self.apply_button)

        #文件树
        self.episode_display = QTextEdit(self)
        self.episode_display.setReadOnly(True)
        self.episode_display.setMinimumHeight(200)
        self.main_layout.addWidget(QLabel("生成的 Episode 文件树"))
        self.main_layout.addWidget(self.episode_display)


        #弹性空间
        self.main_layout.addStretch()
        
        self.current_sensor_data = None
        self.canvas = None

    def display_episode_tree(self, json_path):
        components = parse_robot_config_to_episode_components(json_path)
        # 不实际创建文件，只获取树状文本

        buf = io.StringIO()
        # 临时重定向 stdout
        old_stdout = sys.stdout
        sys.stdout = buf
        generate_episode_structure(components, create_files=False)
        sys.stdout = old_stdout

        tree_text = buf.getvalue()
        self.episode_display.setText(tree_text)

    def clear_properties(self):
        while self.form_layout.count():
            child = self.form_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.current_sensor_data = None
        self.title_label.setText("传感器属性")

    def show_sensor_properties(self, sensor_data):  
        self.clear_properties()  
        self.current_sensor_data = sensor_data  

        # 显示传感器类型  
        self.title_label.setText(f"传感器属性: {sensor_data['type']}")  

        # 显示传感器 ID  
        id_edit = QLineEdit(sensor_data.get('id', ''))  
        id_edit.textChanged.connect(self.update_id)  
        self.form_layout.addRow("ID:", id_edit)  

        comp_type = sensor_data["type"]
        comp_info = self.canvas.component_manager.get_flat_components().get(comp_type, {})
        default_params = comp_info.get("default_params", {}) 
        param_options = comp_info.get("param_options", {})  # 从 yaml 读取 options
        params = default_params.copy()
        params.update(sensor_data.get('params', {}))
        # sensor_data["params"] = params

        # ------------------- 先处理所有普通参数（含 YAML 自带的 msgs_data_floors） -------------------
        for key, value in params.items():  
            # 跳过 ros2_joints 的 'msgs' 参数（后续单独处理，避免重复）
            if key == 'msgs' and comp_type.startswith('arm/ros2_joints') or comp_type.startswith('arm/ros1_joints'):
                continue

            if key == 'output':
                available_outputs = default_params.get("output", [])
                if available_outputs and isinstance(value, list):
                    for option in available_outputs:
                        checkbox = QCheckBox(option)
                        checkbox.setChecked(option in value)
                        checkbox.stateChanged.connect(
                            lambda state, opt=option: self.update_output(opt, state)
                        )
                        self.form_layout.addRow(f"Output {option}:", checkbox)
                else:
                    line_edit = QLineEdit(", ".join(value) if isinstance(value, list) else str(value))
                    line_edit.textChanged.connect(lambda text, k=key: self.update_param(k, text))
                    self.form_layout.addRow("output:", line_edit)

            # 处理有下拉选项的参数
            elif key in param_options:
                options = param_options[key]
                combo = QComboBox()
                combo.setEditable(True)
                combo.addItems(options)
                combo.setCurrentText(str(value) if value is not None else "")
                combo.currentTextChanged.connect(
                    lambda text, k=key: self.update_param(k, text)
                )
                self.form_layout.addRow(f"{key}:", combo)

            elif isinstance(value, bool):  
                checkbox = QCheckBox()  
                checkbox.setChecked(value)  
                checkbox.stateChanged.connect(  
                    lambda state, k=key: self.update_param(k, state == Qt.Checked)  
                )  
                self.form_layout.addRow(f"{key}:", checkbox)  

            elif isinstance(value, int):  
                spinbox = QSpinBox()  
                spinbox.setRange(0, 99999)
                spinbox.setValue(value)  
                spinbox.valueChanged.connect(  
                    lambda val, k=key: self.update_param(k, val)  
                )  
                self.form_layout.addRow(f"{key}:", spinbox)  

            elif isinstance(value, float):  
                spinbox = QDoubleSpinBox()  
                spinbox.setRange(-9999.0, 9999.0)
                spinbox.setValue(value)  
                spinbox.valueChanged.connect(  
                    lambda val, k=key: self.update_param(k, val)  
                )  
                self.form_layout.addRow(f"{key}:", spinbox)  

            else:  
                # 直接渲染 YAML 中的 msgs_data_floors（无重复）
                line_edit = QLineEdit(str(value))  
                line_edit.textChanged.connect(  
                    lambda text, k=key: self.update_param(k, text)  
                )
                # 对 msgs_data_floors 加提示
                if key == "msgs_data_floors":
                    self.form_layout.addRow(f"{key}（自定义消息专用）：", line_edit)
                else:
                    self.form_layout.addRow(f"{key}:", line_edit)

        # ------------------- 单独处理 ros2_joints 的 msgs 参数 -------------------
        if comp_type.startswith('arm/ros2_joints') or  comp_type.startswith('arm/ros1_joints'):
            # 固定默认消息类型为 sensor_msgs/JointState
            fixed_default_msg = "sensor_msgs/JointState"
            current_msg_type = params.get("msgs", fixed_default_msg)
            current_data_floors = params.get("msgs_data_floors", "")

            # 强制默认消息类型的参数值
            if current_msg_type == fixed_default_msg:
                self.current_sensor_data["params"]["msgs"] = fixed_default_msg
                self.current_sensor_data["params"]["msgs_data_floors"] = ""  # 默认消息强制清空

            # 显示消息类型（默认/自定义区分显示）
            if current_msg_type == fixed_default_msg:
                # 默认消息：不可修改的标签
                msg_label = QLabel(fixed_default_msg)
                msg_label.setStyleSheet("color: #333; background-color: #f0f0f0; padding: 3px 8px; border-radius: 3px;")
                self.form_layout.addRow("msgs（默认）：", msg_label)
            else:
                # 自定义消息：显示当前类型（不可直接编辑）
                msg_label = QLabel(current_msg_type)
                msg_label.setStyleSheet("color: #d9534f; background-color: #fef7fb; padding: 3px 8px; border-radius: 3px;")
                self.form_layout.addRow("msgs（自定义）：", msg_label)

            # 自定义消息按钮
            def on_custom_msg_click():
                nonlocal current_msg_type, current_data_floors
                # 移除原有的固定 ROS2 工具类导入，改为由弹窗内部动态处理
                # 传递开发模式到弹窗（通过 parent 参数自动传递）
                existing_config = {
                    "msgs": current_msg_type,
                    "msgs_data_floors": current_data_floors
                }
                dialog = ROS2CustomMsgConfigDialog(existing_config, parent=self)
                if dialog.exec_():
                    result = dialog.get_result()
                    custom_msg_type = result["msg_type"]
                    custom_data_floors = result["msg_data_floors"]
                    
                    # 校验输入
                    if not custom_msg_type:
                        QMessageBox.warning(self, "输入无效", "自定义消息类型不能为空！")
                        return
                    if '/' not in custom_msg_type:
                        QMessageBox.warning(self, "格式错误", "消息类型格式应为：包名/消息名（如：my_package/MyJointMsg）")
                        return
                    
                    # 更新参数：覆盖消息类型和数据层级
                    self.current_sensor_data["params"]["msgs"] = custom_msg_type
                    self.current_sensor_data["params"]["msgs_data_floors"] = custom_data_floors
                    
                    # 刷新属性面板
                    self.show_sensor_properties(self.current_sensor_data)

            custom_btn = QPushButton("📝 自定义消息类型")
            custom_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4a90e2;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #357abd;
                }
            """)
            custom_btn.clicked.connect(on_custom_msg_click)
            self.form_layout.addRow("", custom_btn)

            # ------------------- 配置状态显示（修复显示不全） -------------------
            if current_msg_type == fixed_default_msg:
                status_text = (
                    "当前配置：默认消息\n"
                    "消息类型：sensor_msgs/JointState\n"
                    "消息数据层级：空（默认）"
                )
                status_style = """
                    color: #0066cc;
                    background-color: #f0f8ff;
                    padding: 8px;
                    border: 1px solid #cce5ff;
                    border-radius: 4px;
                    max-width: 280px;  /* 适配属性面板宽度，留足边距 */
                    white-space: pre-line;  /* 按换行符换行，忽略多余空格 */
                """
            else:
                status_text = (
                    f"当前配置：自定义消息\n"
                    f"消息类型：{current_msg_type}\n"
                    f"消息数据层级：{current_data_floors or '未设置'}"
                )
                status_style = """
                    color: #228B22;
                    background-color: #f8fff8;
                    padding: 8px;
                    border: 1px solid #d4edda;
                    border-radius: 4px;
                    max-width: 280px;  /* 适配属性面板宽度，留足边距 */
                    white-space: pre-line;  /* 按换行符换行，忽略多余空格 */
                """

            # 1. 创建标签并设置核心属性
            status_label = QLabel(status_text)
            status_label.setWordWrap(True)
            status_label.setStyleSheet(status_style)
            status_label.setMinimumHeight(80)  # 预留足够垂直空间，避免挤压
            status_label.setAlignment(Qt.AlignTop)  # 文本顶部对齐，不浪费空间

            # 2. 用 QScrollArea 包裹标签，启用垂直滚动（文本过长时可滚动查看）
            scroll_area = QScrollArea()
            scroll_area.setWidget(status_label)
            scroll_area.setWidgetResizable(True)  # 滚动区域自适应标签大小
            scroll_area.setMaximumHeight(120)  # 限制滚动区域最大高度，避免占用过多面板空间
            scroll_area.setStyleSheet("border: none;")  # 隐藏滚动区域边框，保持样式统一

            # 3. 将滚动区域添加到表单布局
            self.form_layout.addRow("配置状态：", scroll_area)


    def update_output(self, option, state):
        if not self.current_sensor_data:
            return
        outputs = self.current_sensor_data["params"].get("output", [])
        if state == Qt.Checked:
            if option not in outputs:
                outputs.append(option)
        else:
            if option in outputs:
                outputs.remove(option)
        self.current_sensor_data["params"]["output"] = outputs

    def update_param(self, key, value):
        if self.current_sensor_data:
            # 处理 joint_index 列表参数（支持逗号分隔字符串转换）
            if key == "joint_index" and isinstance(value, str):
                try:
                    joint_index = ast.literal_eval(value)
                    value = [int(item.strip()) for item in joint_index.split(",") if item.strip().isdigit()]
                except:
                    pass  # 转换失败保留原始值
            self.current_sensor_data["params"][key] = value

    def update_id(self, value):
        if self.current_sensor_data:
            self.current_sensor_data["id"] = value

    def load_config(self):
        robot_name = self.editable_combo.currentText().strip()  # 获取下拉框的文本（输入/选择的内容）
        
        # ../config/ 表示当前程序所在目录的上一级目录下的 config 文件夹
        config_dir = os.path.join(os.path.pardir, "config")  # 等价于 "../config"
        config_file = os.path.join(config_dir, f"{robot_name}.json")  # 完整路径：../config/xxx.json

        if not robot_name:
            msg = QMessageBox(
                QMessageBox.Warning,
                "输入错误",
                "机器人名称不能为空！",
                parent=self
            )
            msg.setWindowModality(Qt.WindowModal)
            msg.exec_()
            return

        if not os.path.exists(config_file):
            # 文件不存在弹窗
            msg = QMessageBox(
                QMessageBox.Critical,
                "文件不存在",
                f"未找到配置文件！\n路径：{os.path.abspath(config_file)}",
                parent=self  
            )
            msg.setWindowModality(Qt.WindowModal)  # 模态弹窗，优先显示
            msg.exec_()  # 阻塞式显示
            return
        # ------------------- 核心：加载 JSON 并还原组件 -------------------
        try:
            # 读取 JSON 配置文件
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            # 提取组件列表（JSON 中需包含 "components" 字段，与 save_config 对应）
            components = config_data.get("components", [])

            #只支持ros2
            # 提取 develop_mode（默认值为 "dora"，兼容无该字段的旧配置）
            # develop_mode = config_data.get("develop_mode", "dora")
            # # 同步切换组件管理器和左侧下拉框的模式
            # self.canvas.switch_develop_mode(develop_mode)  # 新增方法：统一处理模式切换
            self.canvas.switch_develop_mode("ros2") 

            if not components:
                msg = QMessageBox(
                    QMessageBox.Warning,
                    "无组件配置",
                    "配置文件中未找到组件信息！",
                    parent=self
                )
                msg.setWindowModality(Qt.WindowModal)  # 模态弹窗，优先显示
                msg.exec_()  # 阻塞式显示
                return

            # 1. 清空画布现有组件（避免重复）
            self.canvas.clear_canvas()  # 需在 RobotCanvas 中添加 clear_canvas 方法

            # 2. 按 JSON 配置逐一添加组件
            for comp in components:
                comp_type = comp.get("type")
                comp_params = comp.get("params", {})
                comp_id = comp.get("id")

                if not comp_type:
                    print(f"[警告] 跳过无效组件（缺少 type）：{comp}")
                    continue

                # 调用画布添加组件，并覆盖默认参数和 ID
                self.canvas.add_component_with_config(comp_type, comp_params, comp_id)

            # 3. 同步更新 use_videos 复选框状态（与 JSON 一致）
            use_videos = config_data.get("use_videos", False)
            self.use_videos_checkbox.setChecked(use_videos)

            # （可选）更新 Episode 文件树显示
            self.display_episode_tree(config_file)

        except json.JSONDecodeError as e:
            msg = QMessageBox(
                QMessageBox.Critical,
                "JSON 解析错误",
                f"配置文件格式错误：{str(e)}",
                parent=self
            )
            msg.setWindowModality(Qt.WindowModal)  # 模态弹窗，优先显示
            msg.exec_()  # 阻塞式显示
        except Exception as e:
            msg = QMessageBox(
                QMessageBox.Critical,
                "加载失败",
                f"加载组件时出错：{str(e)}",
                parent=self
            )
            msg.setWindowModality(Qt.WindowModal)  # 模态弹窗，优先显示
            msg.exec_()  # 阻塞式显示

    def save_config(self):
        if not self.canvas:
            # 错误弹窗
            msg = QMessageBox(
                QMessageBox.Critical,
                "保存失败",
                "画布未初始化，无法保存配置！",
                parent=self
            )
            msg.setWindowModality(Qt.WindowModal)
            msg.exec_()
            return
        
        try:
            mode = self.canvas.component_manager.mode
            components_data = self.canvas.items_map
            use_videos = self.use_videos_checkbox.isChecked()
            filename = self.editable_combo.currentText().strip() or "robot_config"
            
            # 检查文件名是否合法（避免特殊字符）
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            if any(char in filename for char in invalid_chars):
                msg = QMessageBox(
                    QMessageBox.Warning,
                    "文件名无效",
                    "文件名包含非法字符（/:*?\"<>|），请修改后重试！",
                    parent=self
                )
                msg.setWindowModality(Qt.WindowModal)
                msg.exec_()
                return
            
            # ------------------- 新增：二次确认弹窗 -------------------
            confirm_msg = f"当前机器人名称：{filename}\n\n是否确认保存配置？\n（配置文件将保存为：{filename}.json）"
            reply = QMessageBox.question(
                self,
                "确认保存",
                confirm_msg,
                QMessageBox.Yes | QMessageBox.No,  # 提供「是」「否」两个选项
                QMessageBox.No  # 默认选中「否」，防止误操作
            )
            if reply != QMessageBox.Yes:
                print(f"[保存配置] 用户取消了保存（机器人名称：{filename}）")
                return
            
            save_path = generate_config(
                components=components_data,
                component_manager=self.canvas.component_manager,
                develop_mode=mode,
                use_videos=use_videos,
                filename=filename
            )
            
            # 生成 episode
            episode_generator.main(save_path)
            
            # 显示文件树
            self.display_episode_tree(save_path)
            
            # 成功弹窗（显示保存路径）
            msg = QMessageBox(
                QMessageBox.Information,
                "保存成功",
                f"配置文件已保存至：\n{os.path.abspath(save_path)}\n\nEpisode 结构已生成并显示！",
                parent=self
            )
            msg.setWindowModality(Qt.WindowModal)
            msg.exec_()
            print(f"[保存配置] 已生成配置文件: {save_path}")

        except subprocess.CalledProcessError as e:
            # episode 生成失败
            msg = QMessageBox(
                QMessageBox.Critical,
                "保存失败",
                f"配置文件保存成功，但生成 Episode 失败：\n{str(e)}",
                parent=self
            )
            msg.setWindowModality(Qt.WindowModal)
            msg.exec_()
            print(f"[错误] 调用 episode_generator.py 失败: {e}")
        except Exception as e:
            # 其他保存错误
            msg = QMessageBox(
                QMessageBox.Critical,
                "保存失败",
                f"保存配置时发生错误：\n{str(e)}",
                parent=self
            )
            msg.setWindowModality(Qt.WindowModal)
            msg.exec_()
            print(f"[错误] 保存配置失败: {e}")

    def apply_config(self):
        if not self.canvas:
            msg = QMessageBox(
                QMessageBox.Critical,
                "应用失败",
                "画布未初始化，无法应用配置！",
                parent=self
            )
            msg.setWindowModality(Qt.WindowModal)
            msg.exec_()
            return
        
        try:
            mode = self.canvas.component_manager.mode
            filename = self.editable_combo.currentText().strip() or "robot_config"
            json_path = f"../config/{filename}.json"
            
            # 检查配置文件是否存在
            if not os.path.exists(json_path):
                msg = QMessageBox(
                    QMessageBox.Warning,
                    "文件不存在",
                    f"未找到配置文件：\n{os.path.abspath(json_path)}\n\n请先保存配置！",
                    parent=self
                )
                msg.setWindowModality(Qt.WindowModal)
                msg.exec_()
                return
            
            # ------------------- 新增：二次确认弹窗 -------------------
            confirm_msg = f"当前机器人名称：{filename}\n当前模式：{mode}\n\n是否确认应用配置？\n（将基于 {filename}.json 生成对应模式的配置脚本）"
            reply = QMessageBox.question(
                self,
                "确认应用",
                confirm_msg,
                QMessageBox.Yes | QMessageBox.No,  # 提供「是」「否」两个选项
                QMessageBox.No  # 默认选中「否」，防止误操作
            )
            if reply != QMessageBox.Yes:
                print(f"[应用配置] 用户取消了应用（机器人名称：{filename}，模式：{mode}）")
                return
            

            current_dir = os.path.dirname(os.path.abspath(__file__))
            success_msg = f"已成功应用配置（{mode} 模式）！\n配置文件：{os.path.basename(json_path)}"

            if mode == "dora":
                generate_dora.main(filename)
            elif mode == "ros1":
                generate_ros1.main(filename)
            if mode == "ros2":
                generate_ros2_aio.main(filename)
            else:
                raise ValueError(f"未知模式: {mode}")
            
            # 应用成功弹窗
            msg = QMessageBox(
                QMessageBox.Information,
                "应用成功",
                success_msg,
                parent=self
            )
            msg.setWindowModality(Qt.WindowModal)
            msg.exec_()
            print(f"[应用配置] 已调用 generate_{mode}.py 处理 {json_path}")

        except ValueError as e:
            # 模式错误
            msg = QMessageBox(
                QMessageBox.Critical,
                "应用失败",
                str(e),
                parent=self
            )
            msg.setWindowModality(Qt.WindowModal)
            msg.exec_()
        except subprocess.CalledProcessError as e:
            # 脚本调用失败
            msg = QMessageBox(
                QMessageBox.Critical,
                "应用失败",
                f"调用生成脚本失败：\n{str(e)}",
                parent=self
            )
            msg.setWindowModality(Qt.WindowModal)
            msg.exec_()
            print(f"[错误] 调用 generate_{mode}.py 失败: {e}")
        except Exception as e:
            # 其他错误
            msg = QMessageBox(
                QMessageBox.Critical,
                "应用失败",
                f"应用配置时发生错误：\n{str(e)}",
                parent=self
            )
            msg.setWindowModality(Qt.WindowModal)
            msg.exec_()
            print(f"[错误] 应用配置失败: {e}")

# 中间画布
class RobotCanvas(QGraphicsView):
    def __init__(self, property_panel, component_manager,component_list=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.property_panel = property_panel
        self.component_manager = component_manager
        self.items_map = []
        self.component_list = component_list  
        self.item_size = (200, 150)
        self.columns = 1
        self.margin = 20

    def show_temporary_message(self, text, duration=2000):
        """显示临时提示文字（自动消失）"""
        label = QLabel(text, self)
        label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                padding: 8px 15px;
                border-radius: 8px;
                font-size: 14px;
            }
        """)
        label.adjustSize()
        label.move(
            (self.width() - label.width()) // 2,
            20
        )
        label.show()

        QTimer.singleShot(duration, label.deleteLater)


    def add_component(self, comp_type):
        comp_info = self.component_manager.get_flat_components()[comp_type]
        default_params = comp_info.get("default_params", {})

        def on_delete():
            if group == "camera" and hasattr(widget, "camera_name"):
                detect_components.release_camera(widget.camera_name)
            self._scene.removeItem(proxy)
            self.items_map = [it for it in self.items_map if it["proxy"] != proxy]
            self.property_panel.clear_properties()
            self.relayout_items()
            # self.property_panel.canvas.component_list.refresh_components()

        group, name = comp_type.split("/", 1)
        if group == "camera":
            available_cams = detect_components.get_available_cameras(mode=self.component_manager.mode)
            print(f"Available cameras: {available_cams}")  # 调试输出

            matched_cam = available_cams[0] if available_cams else None
            print(f"Selected camera: {matched_cam}")  # 调试输出
            
            if matched_cam:
                success = detect_components.allocate_camera(matched_cam, comp_type)
                if success:
                    widget = CameraWidget(matched_cam, on_delete_callback=on_delete)
                else:
                    widget = QLabel("摄像头被占用")
            else:
                self.show_temporary_message("当前没有可用的摄像头！")
                return
        else:
            widget = ComponentItemWidget(comp_type, on_delete)


        proxy = QGraphicsProxyWidget()
        proxy.setWidget(widget)
        proxy.setFlag(QGraphicsProxyWidget.ItemIsSelectable, True)
        proxy.setZValue(10)

        data = {
            "type": comp_type,
            "params": default_params.copy(),
            "id": str(uuid.uuid4())
        }
        proxy.sensor_data = data
        self.items_map.append({"proxy": proxy, "data": data})
        self._scene.addItem(proxy)
        self.relayout_items()
        # self.property_panel.canvas.component_list.refresh_components()

    def relayout_items(self):
        for idx, item in enumerate(self.items_map):
            row = idx // self.columns
            col = idx % self.columns
            x = col * (self.item_size[0] + self.margin)
            y = row * (self.item_size[1] + self.margin)
            item["proxy"].setPos(x, y)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        item = self.itemAt(event.pos())
        if isinstance(item, QGraphicsProxyWidget) and hasattr(item, "sensor_data"):
            self.property_panel.show_sensor_properties(item.sensor_data)

    def clear_canvas(self):
        """清空画布所有组件（含摄像头释放）"""
        # 释放摄像头资源
        for item in self.items_map:
            proxy = item["proxy"]
            widget = proxy.widget()
            # 处理摄像头组件的释放
            if hasattr(widget, "camera_name"):
                detect_components.release_camera(widget.camera_name)
            # 从场景中移除组件
            self._scene.removeItem(proxy)
        # 清空组件列表
        self.items_map.clear()
        # 清空属性面板
        self.property_panel.clear_properties()

    def add_component_with_config(self, comp_type, custom_params, custom_id):
        """按自定义配置添加组件（同步 msg_data_floors 配置）"""
        comp_info = self.component_manager.get_flat_components().get(comp_type)
        if not comp_info:
            print(f"[警告] 未找到组件类型：{comp_type}")
            return

        default_params = comp_info.get("default_params", {})
        final_params = {**default_params, **custom_params}  # 合并所有参数（含 msg_data_floors）

        # 处理默认消息类型的参数强制规则
        fixed_default_msg = "sensor_msgs/JointState"
        if comp_type.startswith('arm/ros2_joints'):
            current_msg_type = final_params.get("msgs", fixed_default_msg)
            current_msg_path = final_params.get("msg_path", "")
            # 默认消息强制清空 msg_data_floors
            if current_msg_type == fixed_default_msg and not current_msg_path:
                final_params["msgs_data_floors"] = ""

        def on_delete():
            if hasattr(widget, "camera_name") and widget.camera_name:
                detect_components.release_camera(widget.camera_name)
            self._scene.removeItem(proxy)
            self.items_map = [it for it in self.items_map if it["proxy"] != proxy]
            self.property_panel.clear_properties()
            self.relayout_items()

        group, name = comp_type.split("/", 1)
        proxy = QGraphicsProxyWidget()
        proxy.setFlag(QGraphicsProxyWidget.ItemIsSelectable, True)
        proxy.setZValue(10)
        widget = None

        if group == "camera":
            # 原有摄像头组件逻辑不变...
            available_cams = detect_components.get_available_cameras(mode=self.component_manager.mode)
            cam_name = final_params.get("camera_name")
            if cam_name and cam_name in available_cams:
                success = detect_components.allocate_camera(cam_name, comp_type)
                if success:
                    widget = CameraWidget(cam_name, on_delete_callback=on_delete, parent=proxy.widget())
                    widget.camera_name = cam_name
                else:
                    widget = ComponentItemWidget(comp_type, on_delete, parent=proxy.widget())
            else:
                widget = ComponentItemWidget(comp_type, on_delete, parent=proxy.widget())
        else:
            widget = ComponentItemWidget(comp_type, on_delete, parent=proxy.widget())

        widget.setFixedSize(self.item_size[0], self.item_size[1])
        proxy.setWidget(widget)

        # 保存完整参数（含 msg_data_floors）
        data = {
            "type": comp_type,
            "params": final_params,
            "id": custom_id or str(uuid.uuid4())
        }
        proxy.sensor_data = data
        self.items_map.append({"proxy": proxy, "data": data})
        self._scene.addItem(proxy)
        self.relayout_items()

    # ------------------- 新增方法：切换开发模式（ros/dora） -------------------
    def switch_develop_mode(self, mode):
        # 1. 验证模式有效性
        if mode not in ["dora", "ros1", "ros2"]:
            print(f"[WARN] 无效的 develop_mode：{mode}，默认使用 dora")
            mode = "dora"
        
        # 2. 更新组件管理器的模式
        self.component_manager.set_mode(mode)
        
        # 3. 更新左侧组件列表的下拉框选择（同步视觉显示）
        self.component_list.mode_selector.setCurrentText(mode)
        
        # 4. 刷新左侧组件列表（根据新模式显示对应组件）
        self.component_list.refresh_components()
        
        print(f"[INFO] 已自动切换开发模式为：{mode}")

# 左侧组件库 
class ComponentListPanel(QWidget):
    def __init__(self, component_manager, canvas, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.component_manager = component_manager
        self.canvas = canvas

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)
        self.setMaximumWidth(250)

        self.mode_selector = QComboBox(self)
        # self.mode_selector.addItems(["dora", "ros1","ros2"])
        #只是保留ros2
        self.mode_selector.addItems(["ros2"])
        self.mode_selector.currentTextChanged.connect(self.on_mode_changed)
        self.layout.addWidget(QLabel("选择模式"))
        self.layout.addWidget(self.mode_selector)

        self.items_container = QVBoxLayout()
        self.layout.addLayout(self.items_container)
        self.layout.addStretch()

        self.refresh_components()

    def on_mode_changed(self, mode):
        if self.canvas:  # 防止画布未初始化的异常
            self.canvas.clear_canvas()
        if self.canvas and self.canvas.property_panel:
            self.canvas.property_panel.episode_display.clear()
        self.component_manager.set_mode(mode)
        self.refresh_components()

    def refresh_components(self):
        while self.items_container.count():
            child = self.items_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        components = self.component_manager.get_flat_components()
        available_cams = detect_components.get_available_cameras(mode=self.component_manager.mode)

        for comp_type, comp_info in components.items():
            group, name = comp_type.split("/", 1)
            if group == "camera" and not any(name.lower() in cam.lower() for cam in available_cams):
                continue
            item_widget = QFrame(self)
            item_layout = QHBoxLayout(item_widget)
            name_label = QLabel(comp_type, item_widget)
            item_layout.addWidget(name_label)
            add_button = QPushButton("+", item_widget)
            add_button.setFixedSize(30, 30)
            add_button.clicked.connect(lambda _, t=comp_type: self.canvas.add_component(t))
            item_layout.addWidget(add_button)
            self.items_container.addWidget(item_widget)

# 主窗口
class RobotConfigWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #设置画布大小及标题
        self.resize(1300, 800)
        self.setWindowTitle("机器人传感器配置工具")
        #设置主要布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 1. 初始化组件管理器（最先创建，无依赖）
        self.component_manager = ComponentManager("../config/components.yaml")

        # 2. 创建属性面板（无依赖）
        self.property_panel = PropertyPanel(self)

        # 3. 创建左侧组件列表（依赖 component_manager，暂时不依赖 canvas）
        self.component_list = ComponentListPanel(self.component_manager, None)

        # 4. 创建画布（依赖 property_panel、component_manager、component_list）
        self.canvas = RobotCanvas(
            self.property_panel,
            self.component_manager,
            self.component_list  # 关键：传递左侧组件列表引用
        )

        # 5. 补充赋值依赖（画布给属性面板，画布给左侧组件列表）
        self.property_panel.canvas = self.canvas
        self.component_list.canvas = self.canvas  # 左侧组件列表需要画布引用用于添加组件

        # 添加组件到布局
        layout.addWidget(self.component_list)
        layout.addWidget(self.canvas)
        layout.addWidget(self.property_panel)

        

# ------------------ 启动 ------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotConfigWindow()
    window.show()
    sys.exit(app.exec_())