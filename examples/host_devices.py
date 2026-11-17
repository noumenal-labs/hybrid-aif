"""The device flag of an example that runs replicas by one batched ``run`` on a CPU host.

``run(..., batch=B)`` runs its replicas in parallel across devices (each device runs the whole
scan over its share, design batched-run.md, Phase C), and a CPU host has one device unless
``XLA_FLAGS=--xla_force_host_platform_device_count=N`` is set before jax is imported: on one
device a batched run is no faster than its replicas one after another (the survey rover's step
is compute-bound). An example run as a script calls :func:`use_host_devices` before it imports
jax; it sets the flag when it is unset and the run is batched, one device per core up to the
replicas, and says so in one line.
"""

import os
import sys


def replicas_in_argv(flag: str = "--replicas", argv: list[str] | None = None) -> int:
    """The value of ``flag`` on the command line (``--replicas 8`` or ``--replicas=8``), read
    before ``argparse`` runs (jax must not be imported yet); 1 where it is absent."""
    argv = sys.argv[1:] if argv is None else argv
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.startswith(f"{flag}="):
            return int(arg.split("=", 1)[1])
    return 1


def use_host_devices(replicas: int) -> int | None:
    """Give the CPU host ``min(os.cpu_count(), replicas)`` devices for a batched run of
    ``replicas``, by ``XLA_FLAGS``, when the flag is unset and jax is not yet imported; print
    one line saying so. Returns the device count set, or ``None`` where nothing was set."""
    if replicas < 2 or "XLA_FLAGS" in os.environ:
        return None
    count = min(os.cpu_count() or 1, replicas)
    if count < 2:
        return None
    if "jax" in sys.modules:
        print(
            f"note: jax is already imported, so the {replicas} replicas run on one device; set "
            f"XLA_FLAGS=--xla_force_host_platform_device_count={count} before starting Python "
            "to run them in parallel",
            flush=True,
        )
        return None
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={count}"
    print(
        f"note: XLA_FLAGS was unset; the {replicas} replicas run over {count} host devices "
        f"(--xla_force_host_platform_device_count={count}), since a CPU host runs batched "
        "replicas in parallel only across devices",
        flush=True,
    )
    return count
