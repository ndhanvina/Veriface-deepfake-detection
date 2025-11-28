import json
import os
from filelock import FileLock

CONFIG_PATH = "config.json"

def load_config(path=CONFIG_PATH):
    """Load configuration from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def safe_load_json(path, default=None):
    """Thread-safe JSON load."""
    if default is None:
        default = {}
    
    if not os.path.exists(path):
        return default

    lock_path = path + ".lock"
    lock = FileLock(lock_path)
    
    with lock:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return default

def safe_save_json(path, data):
    """Thread-safe JSON save."""
    lock_path = path + ".lock"
    lock = FileLock(lock_path)
    
    with lock:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
