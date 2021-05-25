"""
Sample strategy for training supernet
"""

import random
import numpy as np
import pickle

SPACES = (
    [[4, 8, 12, 16]] * 7
    + [[4, 8, 12, 16, 20, 24, 28, 32]] * 6
    + [[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]] * 7
)

class Generator():
    def __init__(self, sampler, maxsize=256, *args, **kwargs):
        self.sampler = sampler
        self.pointer = 0
        self.args = args
        self.kwargs = kwargs
        self.maxsize = maxsize
        self.history = self.sampler(maxsize, *self.args, **self.kwargs)
    
    def __call__(self):
        x = self.history[self.pointer]
        self.pointer += 1
        if self.pointer == len(self.history):
            self.pointer = 0
            self.history = self.sampler(self.maxsize, *self.args, **self.kwargs)
        return x
    
class WeightedSample:
    def __init__(self, archs=None, probs=None):
        if isinstance(archs, str):
            self.archs = pickle.load(open(archs, 'rb'))
        else:
            self.archs = archs
        if probs is None:
            self.probs = [1 / len(archs)] * len(archs)
        elif isinstance(probs, str):
            self.probs = pickle.load(open(probs, 'rb'))
        else:
            self.probs = probs
    
    def sample(self, numbers):
        return random.choices(self.archs, self.probs, k=numbers)

def uniform_sample(numbers, space=SPACES):
    return [[random.choice(x) for x in space] for _ in range(numbers)]

def _strict_fair_one_batch(space=SPACES) -> list:
    layer_list = []
    for i in range(19):
        layer_list.append(
            np.random.choice(space[i] * int(16 / len(space[i])), size=16, replace=False,)
        )
    layer_list = np.array(layer_list)
    return np.transpose(layer_list).tolist()

def strict_fair_sample(numbers, space=SPACES):
    assert numbers % 16 == 0, "sample number should be 16 * n"
    sampled = []
    for _ in range(numbers // 16):
        sampled += _strict_fair_one_batch(space)
    return sampled
