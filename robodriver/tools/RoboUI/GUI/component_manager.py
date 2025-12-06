import yaml

class ComponentManager:
    def __init__(self, yaml_file="../config/components.yaml"):
        with open(yaml_file, "r", encoding="utf-8") as f:
            self.components = yaml.safe_load(f)
        self.mode = "dora"

    def set_mode(self, mode):
        if mode in self.components:
            self.mode = mode
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
