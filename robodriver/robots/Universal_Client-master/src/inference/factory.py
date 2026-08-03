from src.inference.registry import get_engine_class

# 👇 显式导入插件模块，触发里面的 @register_engine 装饰器执行
import src.inference.plugins.BeingH.beingh_plugin
# import src.inference.plugins.octo_plugin  <- 以后有新模型，就加一行 import

def create_inference_engine(config: dict):
    engine_name = config.get('inference', {}).get('engine', 'none').lower()
    
    # 动态从注册表中获取类
    engine_cls = get_engine_class(engine_name)
    
    if engine_cls:
        # 实例化并返回
        return engine_cls(config)
        
    elif engine_name == 'none':
        # 提供一个空跑的 Dummy，防止报错
        class DummyEngine:
            def predict(self, raw_states, instruction): return {}
        return DummyEngine()
        
    else:
        raise ValueError(f"❌ 找不到推理引擎插件 '{engine_name}'。请检查 yaml 配置或是否正确导入了对应的 plugin。")