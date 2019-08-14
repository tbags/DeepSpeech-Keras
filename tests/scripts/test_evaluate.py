import os
from typing import List, Tuple, Iterable
import pytest
import h5py
import pandas as pd
import numpy as np
from keras.models import Model
from source.deepspeech import DeepSpeech
from source.generator import DataGenerator
from source.metric import Metric, get_metrics
from scripts.evaluate import get_activations_function, save_in, evaluate
from source.utils import chdir
is_same = lambda A, B: all(np.array_equal(a, b) for a, b in zip(A, B))

chdir(to='ROOT')
np.random.seed(123)


@pytest.fixture
def batch(generator: DataGenerator) -> Tuple[np.ndarray, np.ndarray]:
    inputs, targets = generator[0]
    X, y = inputs['X'], targets['main_output']
    return X, y


def test_get_activations_function(model: Model, batch: Tuple[np.ndarray, np.ndarray]):
    X, y = batch
    get_activations = get_activations_function(model)
    *activations, y_hat = get_activations([X, 0])
    assert len(activations) + 1 == sum(True for layer in model.layers if layer.weights)     # Activations and prediction
    batch, time, features = X.shape
    assert activations[0].shape == (batch, time, 1, 64)
    assert activations[1].shape == (batch, time, 64)
    assert y_hat.shape == (2, 299, 36)


@pytest.fixture
def layer_outputs(model: Model, batch: Tuple[np.ndarray, np.ndarray]) -> List[np.ndarray]:
    X, y = batch
    get_activations = get_activations_function(model)
    *activations, y_hat = get_activations([X, 0])
    return [X, *activations, y_hat]


@pytest.fixture
def metrics(deepspeech: DeepSpeech, layer_outputs: List[np.ndarray], batch: Tuple[np.ndarray, np.ndarray]) -> Iterable[Metric]:
    X, y = batch
    y_hat = layer_outputs[-1]
    predict_sentences = deepspeech.decode(y_hat)
    true_sentences = deepspeech.get_transcripts(y)
    return get_metrics(sources=predict_sentences, destinations=true_sentences)


@pytest.fixture
def store_path() -> str:
    return 'tests/evaluation.hdf5'


@pytest.fixture
def references() -> pd.DataFrame:
    return pd.DataFrame(columns=['sample_id', 'transcript', 'prediction', 'wer', 'cer']).set_index('sample_id')


def test_save_in(store_path: str, layer_outputs: List[np.ndarray], metrics: Iterable[Metric], references: pd.DataFrame):
    with h5py.File(store_path, mode='w') as store:
        metrics = list(metrics)
        save_in(store, layer_outputs, metrics, references)
        assert len(references) == 2
        assert all(references.columns.values == np.array(['transcript', 'prediction', 'wer', 'cer']))
        save_in(store, layer_outputs, metrics, references)
        assert len(references) == 4

    sample_id = np.random.choice(references.index)
    with h5py.File(store_path, mode='r') as store:
        output_index = 0
        sample_X = store[f'outputs/{output_index}/{sample_id}']
        assert sample_X.shape == (299, 80)


def test_evaluate(deepspeech: DeepSpeech, generator: Iterable, store_path: str):
    metrics = evaluate(deepspeech, generator, save_activations=True, store_path=store_path)
    with pd.HDFStore(store_path, mode='r') as store:
        references = store['references']

    assert len(references) == len(metrics) == 8
    assert all(references.columns.values == np.array(['transcript', 'prediction', 'wer', 'cer']))
    os.remove(store_path)
