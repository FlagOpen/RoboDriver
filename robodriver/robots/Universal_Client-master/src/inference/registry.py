# 统一管理所有推理引擎的字典
_INFERENCE_ENGINES = {}

def register_engine(name: str):
    """
    类装饰器：用于将具体的推理引擎类注册到系统中
    用法: @register_engine("beingh")
    """
    def decorator(cls):
        _INFERENCE_ENGINES[name.lower()] = cls
        return cls
    return decorator

def get_engine_class(name: str):
    return _INFERENCE_ENGINES.get(name.lower())