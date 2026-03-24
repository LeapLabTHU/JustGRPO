# https://github.com/TIGER-AI-Lab/AceCoder/blob/main/data/inference/EvaluateInferencedCode.py

import os
import shutil
import subprocess
import multiprocessing
from multiprocessing import Pool, TimeoutError
from typing import List, Optional
import numpy as np

# Save original functions to restore after reliability_guard
cwd = os.getcwd()
cache_wd = cwd + "/cache"

tmp_chmod = os.chmod
tmp_fchmod = os.fchmod
tmp_chdir = os.chdir
tmp_rmdir = os.rmdir
tmp_print = print
tmp_rm_tree = shutil.rmtree
tmp_unlink = os.unlink
tmp_replace = os.replace
tmp_getcwd = os.getcwd
tmp_move = shutil.move
tmp_popen = subprocess.Popen
tmp_rename = os.rename


def run_single_test(func: str, test: str) -> int:
    """Run one test case and return 1 if passed, else 0."""
    execution_context = {"__builtins__": __builtins__}
    try:
        exec(func, execution_context)
        exec(test, execution_context)
        return 1
    except Exception:
        return 0


def local_execute(pairs: List[tuple]) -> List[float]:
    """Execute code-test pairs and return pass rates.

    Args:
        pairs: List of (code, test_cases) tuples

    Returns:
        List of pass rates (0.0 - 1.0)
    """
    results = []
    for code, tests in pairs:
        result = get_successful_tests(code, tests)
        results.append(np.mean(result) if result else 0.0)
    return results


def get_successful_tests(
    program: str,
    tests: List[str],
    max_execution_time: float = 1.0,
    num_workers: int = 4
) -> List[int]:
    """Run a program against a list of tests.

    Args:
        program: Python program string
        tests: List of assert statements
        max_execution_time: Timeout per test in seconds
        num_workers: Number of parallel workers

    Returns:
        List of 0/1 indicating passed or not
    """
    test_length = len(tests)
    if test_length == 0:
        return []
    if not should_execute(program=program, tests=tests):
        return [0] * test_length

    reliability_guard()
    results = []

    with Pool(processes=num_workers) as pool:
        async_results = [pool.apply_async(run_single_test, (program, test)) for test in tests]
        for ar in async_results:
            try:
                results.append(ar.get(timeout=max_execution_time))
            except TimeoutError:
                results.append(0)

    partial_undo_reliability_guard()
    return results


def should_execute(program: str, tests: List[str]) -> bool:
    """Determine if we should execute this program for safety reasons."""
    dangerous_commands = [
        "threading",
        "multiprocess",
        "multiprocessing",
        "import os",
        "from os",
        "shutil",
        "import torch",
        "from torch",
        "import sklearn",
        "from sklearn",
    ]
    for comm in dangerous_commands:
        if comm in program:
            return False
    return True


def partial_undo_reliability_guard():
    """Restore functions disabled by reliability_guard."""
    import builtins

    os.chmod = tmp_chmod
    os.fchmod = tmp_fchmod
    os.chdir = tmp_chdir
    os.unlink = tmp_unlink
    os.rmdir = tmp_rmdir
    builtins.print = tmp_print

    os.chdir(cwd)
    shutil.rmtree = tmp_rm_tree
    shutil.move = tmp_move
    subprocess.Popen = tmp_popen
    os.replace = tmp_replace
    os.getcwd = tmp_getcwd
    os.rename = tmp_rename


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """Disable destructive functions for safe code execution.

    WARNING: This is NOT a security sandbox. Use with caution.
    """
    import faulthandler
    import platform

    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None
    builtins.print = lambda *args, **kwargs: None

    os.makedirs(cache_wd, exist_ok=True)
    os.chdir(cache_wd)

    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    subprocess.Popen = None

    import sys
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
