import numpy as np
import tensorflow as tf
from keras.backend import tensorflow_backend as k
from source.deepspeech import DeepSpeech
np.random.seed(123)


def test_adversarial_generator():
    ctc_loss, adversarial_loss = DeepSpeech.get_losses(adversarial=True)

    rich_activations = np.random.normal(0.3, 0.1, size=[100]).reshape([-1, 20])
    syn_activations = np.random.normal(0.5, 0.1, size=[100]).reshape([-1, 20])
    activations = np.concatenate([rich_activations, syn_activations], axis=0)
    labels = np.concatenate([np.zeros(5), np.ones(5)])

    labels_tensor = tf.Variable(labels)
    activations_tensor = tf.Variable(activations)
    loss_tensor = adversarial_loss(labels_tensor, activations_tensor)
    loss = k.eval(loss_tensor)
    assert np.isclose(loss, np.sum(np.abs(rich_activations.mean(axis=0) - syn_activations.mean(axis=0))))
