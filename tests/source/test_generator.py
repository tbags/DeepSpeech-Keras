from copy import deepcopy as copy
import numpy as np
from generator import DistributedDataGenerator
from source.generator import DataGenerator
from source.utils import chdir
is_same = lambda A, B: all(np.array_equal(a, b) for a, b in zip(A, B))
chdir(to='ROOT')


def test_syn_create_generator_from_audio_files(syn_generator: DataGenerator):
    assert len(syn_generator) == 2
    inputs, targets = syn_generator[0]
    X, y = inputs['X'], targets['main_output']
    assert X.shape == (2, 299, 80)
    assert y.shape == (2, 45)
    is_synthesized = targets['is_synthesized']
    assert is_synthesized.shape == (2, 1)
    assert is_synthesized.dtype == float
    assert is_same(is_synthesized, np.array([[1], [1]]))


def test_create_generator_from_audio_files(generator: DataGenerator):
    assert len(generator) == 2
    inputs, targets = generator[0]
    X, y = inputs['X'], targets['main_output']
    assert X.shape == (2, 299, 80)
    assert y.shape == (2, 45)


def test_distributed_generator(generator: DataGenerator):
    generator = DistributedDataGenerator(
        generators=[copy(generator) for i in range(5)]
    )
    assert len(generator) == 10
    assert generator[0]
    assert generator[9]
