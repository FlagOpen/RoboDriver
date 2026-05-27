import yaml
import os

class ConfigLoader:
    def __init__(self, config_relative_path="config/settings.yaml"):
        """
        初始化配置加载器
        :param config_relative_path: 配置文件相对于项目根目录的路径
        """
        self.config_path = config_relative_path
        self.data = None

    def load(self):
        """
        执行加载动作，解析 YAML 并返回字典
        """
        # 1. 获取项目根目录 (假设此文件在 src/utils/ 下，往上找3层是根目录)
        # 如果你的目录结构变了，这里可能需要调整层级
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 2. 拼接完整路径
        full_path = os.path.join(base_dir, self.config_path)
        
        # 3. 检查文件是否存在
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Config file not found at: {full_path}")
            
        # 4. 读取并解析
        with open(full_path, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
            return self.data

    def get(self, key, default=None):
        """
        可选：封装一个安全的获取方法，防止 KeyError
        用法: loader.get('app')
        """
        if self.data is None:
            self.load()
        return self.data.get(key, default)