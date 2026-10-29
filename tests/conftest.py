"""Test-session JAX configuration: a persistent compilation cache under the repo.

Compiled executables are written to ``.jax_cache/`` (git-ignored) so repeated
runs skip XLA compilation of unchanged programs. Override the location with
``HYBRID_AIF_JAX_CACHE``.
"""

import os
from pathlib import Path

import jax

_DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".jax_cache"
_CACHE_DIR = os.environ.get("HYBRID_AIF_JAX_CACHE", str(_DEFAULT_CACHE))

jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
