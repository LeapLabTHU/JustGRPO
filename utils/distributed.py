import builtins
import contextlib

import torch
from accelerate import Accelerator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Global accelerator instance
_ACCELERATOR = None


def init():
    """Initialize the global Accelerator with extended timeout."""
    global _ACCELERATOR
    _ACCELERATOR = Accelerator()
    setup_for_distributed(_ACCELERATOR.process_index == 0)


def get_accelerator():
    """Return the global Accelerator instance."""
    return _ACCELERATOR


def get_rank():
    """Return the global process rank."""
    return _ACCELERATOR.process_index


def get_local_rank():
    """Return the local process rank (within the node)."""
    return _ACCELERATOR.local_process_index


def get_world_size():
    """Return the total number of processes."""
    return _ACCELERATOR.num_processes


def barrier():
    """Synchronize all processes."""
    _ACCELERATOR.wait_for_everyone()


def setup_for_distributed(is_master):
    """
    Monkeypatch builtins.print so that only the master process prints by default.
    Non-master processes can still print with print(..., force=True).
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


@contextlib.contextmanager
def ddp_sync(module, sync):
    """
    Context manager to enable/disable gradient synchronization for DDP/FSDP.
    Used during gradient accumulation: sync=False skips gradient sync until the final step.
    """
    if sync or not isinstance(module, (DDP, FSDP)):
        yield
    else:
        with module.no_sync():
            yield
