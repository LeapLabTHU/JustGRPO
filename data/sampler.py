import torch 
import numpy as np

import utils.distributed as dist
from itertools import islice


class InfiniteSampler(torch.utils.data.Sampler):
    """
    sampler from official edm repo
    """
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=False, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


def split_by_node(src, group=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size > 1:
        yield from islice(src, rank, None, world_size)
    else:
        yield from src
        
def nosplit_by_node(src, group=None):
    yield from src