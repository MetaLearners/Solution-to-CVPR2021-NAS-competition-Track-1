from copy import deepcopy
import os
import paddle
import numpy as np
from paddle.vision.transforms import (
    RandomHorizontalFlip,
    Compose,
    BrightnessTransform,
    ContrastTransform,
    RandomCrop,
    Normalize,
    RandomRotation,
)
from paddle.vision.datasets import Cifar100
import random
from paddle.io import DataLoader
import json

def load(path):
    return json.load(open(path))

def save(obj, path):
    json.dump(obj, open(path, 'w'))

def pad(tensor, channel):
    c = tensor.shape[1]
    if c != channel:
        shape = tensor.shape
        shape[1] = channel - c
        pad = paddle.zeros(shape, dtype='float32')
        return paddle.concat([tensor, pad], axis=1)
    return tensor

def seed_global(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class ToArray(object):
    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.0
        return img.astype("float32")


class RandomApply(object):
    def __init__(self, transform, p=0.5):
        super().__init__()
        self.p = p
        self.transform = transform

    def __call__(self, img):
        if self.p < random.random():
            return img
        img = self.transform(img)
        return img


class Dataset:
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.1942, 0.1918, 0.1958]
    TRANSFORM = Compose(
        [
            RandomCrop(32, padding=4),
            RandomApply(BrightnessTransform(0.1)),
            RandomApply(ContrastTransform(0.1)),
            RandomHorizontalFlip(),
            RandomRotation(15),
            ToArray(),
            Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    VAL_TRANSFORM = Compose([ToArray(), Normalize(CIFAR_MEAN, CIFAR_STD)])

    def __init__(
        self, path=os.path.join('./data', "cifar-100-python.tar.gz"), cache='./data'
    ) -> None:
        self.path = path
        self.cache = cache
        if not os.path.exists(os.path.join(self.cache, 'dset.data')):
            # cache the loaded dataset
            self._parse_dataset()
            self._cache()
        else:
            # load from cache
            self._load_cache()

    def _parse_dataset(self):
        self.train_set = Cifar100(self.path, mode="train", transform=self.TRANSFORM)
        self.test_set = Cifar100(self.path, mode="test", transform=self.VAL_TRANSFORM)

    def get_loader(self, batch_size, mode="train", num_workers=2, shuffle='auto'):
        if shuffle == 'auto':
            shuffle = True if mode == 'train' else False
        elif isinstance(shuffle, str):
            shuffle = eval(shuffle)
        return DataLoader(
            self.train_set if mode == "train" else self.test_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
    
    def _cache(self):
        assert self.cache is not None
        os.makedirs(self.cache, exist_ok=True)
        paddle.save({'train': self.train_set, 'test': self.test_set}, os.path.join(self.cache, 'dset.data'))
    
    def _load_cache(self):
        data = paddle.load(os.path.join(self.cache, 'dset.data'))
        self.train_set = data['train']
        self.test_set = data['test']


def arch2str(arch: list):
    return '-'.join([str(a) for a in arch])

def str2arch(strs: str):
    return [int(a) for a in strs.strip().split("-")]

def get_param(config: list):
    _config = deepcopy(config)
    _config = [3] + _config
    flex_num = 0
    fixed_num = 0
    for i in range(20):
        if i == 19:
            flex_num += _config[19] * 100 + 100
        else:
            flex_num += _config[i] * _config[i + 1] * 3 * 3
            flex_num += 2 * _config[i + 1]
            fixed_num += 2 * _config[i + 1]
            if i in [1, 7, 13] or (i % 2 == 1 and _config[i] != _config[i + 2]):
                flex_num += _config[i] * _config[i + 2]
                flex_num += 2 * _config[i + 2]
                fixed_num += 2 * _config[i + 2]
    return flex_num, flex_num + fixed_num
