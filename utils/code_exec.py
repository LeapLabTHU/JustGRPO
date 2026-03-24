"""Sandbox code execution and pass@k evaluation for HumanEval/MBPP."""

import contextlib
import faulthandler
import gzip
import io
import json
import multiprocessing
import os
import platform
import re
import signal
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


# ---- JSONL I/O ----

def stream_jsonl(path):
    if path.endswith(".gz"):
        with open(path, "rb") as raw:
            with gzip.open(raw, "rt", encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    yield json.loads(line)


def read_problems(path):
    return {item["task_id"]: item for item in stream_jsonl(path)}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


# ---- Pass@k ----

def estimate_pass_at_k(num_samples, num_correct, k):
    def _est(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    if isinstance(num_samples, int):
        num_samples = [num_samples] * len(num_correct)
    return np.array([_est(int(n), int(c), k) for n, c in zip(num_samples, num_correct)])


# ---- Test code assembly ----

def extract_completion_code(completion):
    match = CODE_BLOCK_RE.search(completion)
    return match.group(1).strip() if match else completion


def build_test_code(sample, problems, is_mbpp=False):
    tid = sample["task_id"]
    code = extract_completion_code(sample["completion"])
    p = problems[tid]
    if is_mbpp:
        return code + "\n" + "\n".join(p["test"])
    return p["prompt"] + code + "\n" + p["test"] + "\n" + f"check({p['entry_point']})"


# ---- Sandbox execution ----

class _Timeout(Exception):
    pass


class _WOStringIO(io.StringIO):
    def read(self, *a, **k): raise IOError
    def readline(self, *a, **k): raise IOError
    def readlines(self, *a, **k): raise IOError
    def readable(self, *a, **k): return False


class _StdinRedirect(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def _time_limit(seconds):
    def handler(signum, frame):
        raise _Timeout()
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _swallow_io():
    s = _WOStringIO()
    with contextlib.redirect_stdout(s), contextlib.redirect_stderr(s), _StdinRedirect(s):
        yield


def _reliability_guard():
    faulthandler.disable()
    import builtins
    import shutil
    import subprocess as sp
    builtins.exit = builtins.quit = builtins.help = None
    os.environ["OMP_NUM_THREADS"] = "1"
    def _dis(*a, **k): raise RuntimeError("Restricted")
    for attr in ("system", "kill", "setuid", "fork"):
        if hasattr(os, attr):
            setattr(os, attr, _dis)
    for attr in ("putenv", "remove", "removedirs", "rmdir", "fchdir", "forkpty",
                 "killpg", "rename", "renames", "truncate", "replace", "unlink",
                 "fchmod", "fchown", "chmod", "chown", "chroot",
                 "lchflags", "lchmod", "lchown", "getcwd", "chdir"):
        if hasattr(os, attr):
            setattr(os, attr, None)
    shutil.rmtree = shutil.move = shutil.chown = None
    sp.Popen = None
    import sys
    for m in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[m] = None


def check_correctness(task_id, sample, timeout=3.0, completion_id=None):
    def unsafe_execute():
        import shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            orig_chdir, orig_rmdir, orig_rmtree = os.chdir, os.rmdir, shutil.rmtree
            try:
                _reliability_guard()
                with _swallow_io():
                    with _time_limit(timeout):
                        exec(sample["test_code"], {})
                result.append("passed")
            except _Timeout:
                result.append("timed out")
            except AssertionError:
                result.append("failed: AssertionError")
            except BaseException as e:
                result.append(f"failed: {e}")
            finally:
                os.chdir = orig_chdir
                os.rmdir = orig_rmdir
                shutil.rmtree = orig_rmtree

    manager = multiprocessing.Manager()
    result = manager.list()
    proc = multiprocessing.Process(target=unsafe_execute)
    proc.start()
    proc.join(timeout=timeout + 1)
    if proc.is_alive():
        proc.kill()
    if not result:
        result.append("timed out")
    return {"task_id": task_id, "completion_id": completion_id,
            "result": result[0], "passed": result[0] == "passed"}


# ---- Evaluation orchestrator ----

def evaluate_functional_correctness(input_file, problem_file, is_mbpp=False,
                                    n_workers=8, timeout=3.0, k=(1,)):
    problems = read_problems(problem_file)
    samples = list(stream_jsonl(input_file))
    completion_ids = Counter()
    results = defaultdict(list)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for s in samples:
            tid = s["task_id"]
            cid = s.get("completion_id", completion_ids[tid])
            futures.append(executor.submit(
                check_correctness, task_id=tid,
                sample={"test_code": build_test_code(s, problems, is_mbpp=is_mbpp)},
                timeout=timeout, completion_id=cid,
            ))
            completion_ids[tid] += 1
        for f in as_completed(futures):
            r = f.result()
            results[r["task_id"]].append((r["completion_id"], r))

    totals, correct = [], []
    for tid in problems:
        task_res = results.get(tid, [])
        passed = [row[1]["passed"] for row in task_res]
        totals.append(len(passed))
        correct.append(sum(passed))
    totals, correct = np.array(totals), np.array(correct)

    return {
        f"pass@{v}": float(estimate_pass_at_k(totals, correct, v).mean())
        for v in k if len(totals) and (totals >= v).all()
    }
