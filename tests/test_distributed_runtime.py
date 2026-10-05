"""Optional dependency smoke check; this does not validate GPU kernels or NCCL."""

import importlib.util
import subprocess
import sys

import pytest


def test_distributed_imports_accept_nvidia_without_amd_properties():
    if not all(importlib.util.find_spec(name) for name in ("torch", "xfuser", "yunchang")):
        pytest.skip("Requires the optional GPU and distributed dependencies")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from unittest.mock import patch
from types import SimpleNamespace
import torch
with patch.object(torch.version, 'cuda', '12.6'), \
     patch.object(torch.cuda, 'is_available', return_value=True), \
     patch.object(torch.cuda, 'get_device_name', return_value='NVIDIA H100 NVL'), \
     patch.object(torch.cuda, 'current_device', return_value=0), \
     patch.object(torch.cuda, 'get_device_capability', return_value=(9, 0)), \
     patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(name='NVIDIA H100 NVL', major=9, minor=0)):
    from openspline_server.hardware import validate_distributed
    validate_distributed()
    from openspline_server._vendor.flash_head.src.modules.flash_head_model import SelfAttention
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
    from yunchang import LongContextAttention
    assert issubclass(xFuserLongContextAttention, LongContextAttention)
""",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
