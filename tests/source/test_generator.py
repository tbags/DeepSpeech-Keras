import pytest
import numpy as np

from configuration import DatasetConfiguration
from deepspeech import DeepSpeech
from source.generator import DataGenerator
from source.utils import chdir
is_same = lambda A, B: all(np.array_equal(a, b) for a, b in zip(A, B))
chdir(to='ROOT')
np.random.seed(123)


def test_syn_distributed_balanced_generator(deepspeech: DeepSpeech):
    dependencies = deepspeech.alphabet, deepspeech.features_extractor
    generator = DatasetConfiguration('tests/data/train-dataset.yaml', *dependencies).create_generator()

    inputs, targets = generator[0]
    labels = targets['is_synthesized']
    proportion = sum(labels) / len(labels)
    assert proportion == 0.5

    inputs, targets = generator[1]
    labels = targets['is_synthesized']
    proportion = sum(labels) / len(labels)
    assert proportion == 0.5
    assert len(inputs['X']) == 8
    assert len(generator) == 2              # The same as rich generator
    with pytest.raises(IndexError):
        _ = generator[2]


def test_syn_create_generator_from_audio_files(syn_generator: DataGenerator):
    assert len(syn_generator) == 4
    inputs, targets = syn_generator[0]
    X, y = inputs['X'], targets['main_output']
    assert X.shape == (2, 299, 80)
    assert y.shape == (2, 45)
    is_synthesized = targets['is_synthesized']
    assert is_synthesized.shape == (2,)
    assert is_synthesized.dtype == float
    assert is_same(is_synthesized, np.array([1, 1]))


def test_create_generator_from_audio_files(generator: DataGenerator):
    assert len(generator) == 4
    inputs, targets = generator[0]
    X, y = inputs['X'], targets['main_output']
    assert X.shape == (2, 299, 80)
    assert y.shape == (2, 45)
