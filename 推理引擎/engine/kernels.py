# ─────────────────────────────────────────────────────────────
# CUDA kernel 薄封装:engine 里统一从这里拿 kernel,例如
#   from .kernels import vector_add
# 编译产物 my_kernels.pyd 在项目根目录,这里把项目根加进 sys.path,
# 保证无论从哪个目录运行(项目根 / 推理引擎 / tests)都能 import。
# ─────────────────────────────────────────────────────────────

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 推理引擎/第一周/engine -> 项目根
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import my_kernels  # noqa: E402  my_kernels.pyd

vector_add = my_kernels.vector_add
# 新 kernel 编译好后,在这里加一行:
# rmsnorm = my_kernels.rmsnorm
