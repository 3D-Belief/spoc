from typing import Callable, Dict, Optional, Any
from threading import RLock

_REGISTRY: Dict[str, Callable] = {}
_CONSTANTS: Dict[str, Any] = {}
_LOCK = RLock()

def register_embodied_task(name: Optional[str] = None):
    """Decorator to register a function under a name."""
    def deco(func: Callable):
        key = name or func.__name__
        with _LOCK:
            if key in _REGISTRY:
                raise KeyError(f"Duplicate registration: {key}")
            _REGISTRY[key] = func
        return func
    return deco

def get_embodied_task(name: str) -> Callable:
    return _REGISTRY[name]

def all_embodied_tasks() -> Dict[str, Callable]:
    return dict(_REGISTRY)

def register_constant(name: str, value: Any, *, overwrite: bool = False) -> None:
    with _LOCK:
        if not overwrite and name in _CONSTANTS:
            raise KeyError(f"Duplicate constant: {name}")
        _CONSTANTS[name] = value

def get_constant(name: str) -> Any:
    return _CONSTANTS[name]

def all_constants() -> dict:
    return dict(_CONSTANTS)