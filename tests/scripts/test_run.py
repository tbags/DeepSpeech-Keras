import os
import numpy as np

from configuration import DatasetConfiguration
from deepspeech import DeepSpeech
from utils import chdir
chdir(to='ROOT')
is_same = lambda A, B: all(np.array_equal(a, b) for a, b in zip(A, B))
is_close = lambda A, B: all(np.allclose(a, b, atol=1e-04) for a, b in zip(A, B))


def test_adversarial_generator():
    deepspeech = DeepSpeech.construct('tests/models/adversarial/configuration.yaml',
                                      'tests/models/adversarial/alphabet.txt')
    dependencies = deepspeech.alphabet, deepspeech.features_extractor
    train_generator = DatasetConfiguration('tests/data/train-dataset.yaml', *dependencies).create_generator()
    dev_generator = DatasetConfiguration('tests/data/dev-dataset.yaml', *dependencies).create_generator()

    # Basic usage
    inputs, targets = train_generator[1]
    deepspeech.parallel_model.fit(inputs, targets)

    # Generator
    history_callback = deepspeech.fit(train_generator, dev_generator, epochs=1, shuffle=False)
    metric_names = ['loss', 'main_output_loss', 'is_synthesized_loss', 'is_synthesized_adversarial_loss',
                    'val_loss', 'val_main_output_loss', 'val_is_synthesized_loss']
    assert all(metric in history_callback.history for metric in metric_names)


def test_adversarial_basic():
    base_deepspeech = DeepSpeech.construct('tests/models/base/configuration.yaml',
                                           'tests/models/base/alphabet.txt')
    base_weights = 'weights.hdf5'
    base_deepspeech.save(base_weights)
    deepspeech = DeepSpeech.construct('tests/models/adversarial/configuration.yaml',
                                      'tests/models/adversarial/alphabet.txt')
    X = np.random.rand(10, 100, 80)
    y = {'main_output': np.random.randint(0, 35, size=[10, 20], dtype=np.int32),
         'is_synthesized': np.random.choice([0, 1], size=[10, 1])}
    deepspeech.parallel_model.train_on_batch(X, y)

    assert all(not is_same(base_deepspeech.model.get_layer(name).get_weights(),
                           deepspeech.model.get_layer(name).get_weights())
               for name in ['base_1', 'base_2', 'base_3'])

    deepspeech.load(base_weights)
    assert all(is_same(base_deepspeech.model.get_layer(name).get_weights(),
                       deepspeech.model.get_layer(name).get_weights())
               for name in ['base_1', 'base_2', 'base_3'])


def test_trainable():
    base_deepspeech = DeepSpeech.construct('tests/models/base/configuration.yaml',
                                           'tests/models/base/alphabet.txt')
    base_weights = 'weights.hdf5'
    base_deepspeech.save(base_weights)

    extended_deepspeech = DeepSpeech.construct('tests/models/extended/configuration.yaml',
                                               'tests/models/extended/alphabet.txt')
    assert all(not extended_deepspeech.model.get_layer(name).trainable for name in ['base_1', 'base_2', 'base_3'])
    assert all(not is_same(base_deepspeech.model.get_layer(name).get_weights(),
                           extended_deepspeech.model.get_layer(name).get_weights())
               for name in ['base_1', 'base_2', 'base_3'])

    extended_deepspeech.load(base_weights)
    assert all(not extended_deepspeech.model.get_layer(name).trainable for name in ['base_1', 'base_2', 'base_3'])
    assert all(is_same(base_deepspeech.model.get_layer(name).get_weights(),
                       extended_deepspeech.model.get_layer(name).get_weights())
               for name in ['base_1', 'base_2', 'base_3'])

    for i in range(10):                                                            # Dummy training (10 epochs / 10 batch_size)
        X = np.random.rand(10, 100, 80)
        y = np.random.randint(0, 35, size=[10, 20], dtype=np.int32)
        extended_deepspeech.parallel_model.train_on_batch(X, y)

    assert is_close(extended_deepspeech.model.predict(X), extended_deepspeech.parallel_model.predict(X)), \
        "The results are the same for model and compiled parallel model."
    os.remove(base_weights)
