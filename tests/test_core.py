import pytest
import os
import json
import torch
from src.utils import safe_load_json, safe_save_json, load_config
from src.model import build_model
from filelock import FileLock

def test_safe_json_ops(tmp_path):
    f = tmp_path / "test.json"
    data = {"key": "value"}
    
    # Test save
    safe_save_json(str(f), data)
    assert f.exists()
    
    # Test load
    loaded = safe_load_json(str(f))
    assert loaded == data
    
    # Test default
    assert safe_load_json("nonexistent.json", default={}) == {}

def test_config_loading():
    config = load_config("config.json")
    assert "paths" in config
    assert "heuristics" in config
    assert config["model"]["backbone"] == "resnet18"

def test_model_build():
    model = build_model(backbone="resnet18", num_classes=2)
    assert isinstance(model, torch.nn.Module)
    # Check output shape
    t = torch.randn(1, 3, 224, 224)
    out = model(t)
    assert out.shape == (1, 2)

def test_file_lock_concurrency(tmp_path):
    # Basic check that lock file is created
    f = tmp_path / "lock_test.json"
    lock_path = str(f) + ".lock"
    
    lock = FileLock(lock_path)
    with lock:
        assert os.path.exists(lock_path)
    # Lock file might persist or be deleted depending on platform/implementation, 
    # but we just want to ensure it doesn't crash.
