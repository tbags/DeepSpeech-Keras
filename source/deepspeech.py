import os
import logging
from functools import partial
from typing import List, Callable
import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.utils import multi_gpu_model
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, TerminateOnNaN, LearningRateScheduler, ReduceLROnPlateau, History
from keras.optimizers import Optimizer, SGD, Adam
from keras.backend.tensorflow_backend import _get_available_gpus as get_available_gpus

from source import model as model_module
from source import audio, text, ctc_decoder, configuration, utils
from source.model import Model
from source.text import Alphabet
from source.audio import FeaturesExtractor
from source.callbacks import CustomModelCheckpoint, CustomTensorBoard, CustomEarlyStopping, ResultKeeper
from source.configuration import ModelConfiguration
logger = logging.getLogger('deepspeech')


class DeepSpeech:
    """
    The DeepSpeech class is the project entry point.
    It helps to do the Speech-To-Text analysis easily, e.g.:
        - perform speech-to-text analysis using pre-trained models,
        - tune pre-trained models to your needs,
        - and create new models on your own.
    """
    def __init__(self,
                 model: Model,
                 alphabet: Alphabet,
                 decoder: Callable,
                 features_extractor: FeaturesExtractor,
                 callbacks: List[Callback] = None,
                 gpus: List[str] = None,
                 parallel_model: Model = None):
        """ Private attributes should not be changed. """
        self._model = model
        self._gpus = gpus
        self._parallel_model = parallel_model
        self.alphabet = alphabet
        self.features_extractor = features_extractor
        self.decoder = decoder
        self.callbacks = callbacks

    @property
    def model(self):
        """ Once defined model should not be changed. """
        return self._model

    @property
    def parallel_model(self):
        return self._parallel_model if self._parallel_model else self._model

    @classmethod
    def construct(cls, config_path: str, alphabet_path: str) -> 'DeepSpeech':
        """ Construct DeepSpeech object base on the configuration and the alphabet files. """
        config = ModelConfiguration(config_path)
        model_dir = os.path.dirname(config_path)
        gpus = get_available_gpus()

        deepspeech_model = cls.get_model(is_gpu=len(gpus) > 0, **config.model)
        parallel_model = cls.distribute_model(deepspeech_model, gpus)

        losses = cls.get_losses(adversarial=deepspeech_model.is_adversarial)
        optimizer = cls.get_optimizer(**config.optimizer)
        cls.compile_model(parallel_model, optimizer, losses)        # Set up training for a distributed model.

        alphabet = cls.get_alphabet(alphabet_path)
        decoder = cls.get_decoder(alphabet=alphabet, model=deepspeech_model, **config.decoder)
        features_extractor = cls.get_features_extractor(**config.features_extractor)
        callbacks = cls.get_callbacks(home_dir=model_dir, configurations=config.callbacks, model=deepspeech_model)
        return cls(deepspeech_model, alphabet, decoder, features_extractor, callbacks, gpus, parallel_model)

    def __call__(self, files: List[str]) -> List[str]:
        """ Easy interaction with the trained model """
        X = self.get_features(files)
        y_hat = self.predict(X)
        sentences = self.decode(y_hat)
        return sentences

    def get_features(self, files: List[str]) -> np.ndarray:
        """ Extract features from files. """
        return self.features_extractor.get_features(files)

    def get_labels(self, transcripts: List[str]) -> np.ndarray:
        """ Convert transcripts to labels. """
        return self.alphabet.get_batch_labels(transcripts)

    def get_transcripts(self, labels: np.ndarray) -> List[str]:
        """ Convert labels to transcripts. """
        return self.alphabet.get_batch_transcripts(labels)

    def fit(self, train_generator, dev_generator, **kwargs) -> History:
        """ Train model using train and dev data generators base on the Keras method."""
        return self.parallel_model.fit_generator(generator=train_generator, validation_data=dev_generator,
                                                 callbacks=self.callbacks, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Predict on the batch. """
        return self.parallel_model.predict_on_batch(X)

    def decode(self, y_hat: np.ndarray) -> List[str]:
        """ Decode probabilities along characters using the decoder (beam search variants). """
        return self.decoder(y_hat)

    def save(self, file_path: str) -> None:
        """ Save model weights. Object can be easily reinitialized. """
        self.model.save_weights(file_path)

    @utils.pretrained_models
    def load(self, file_path: str) -> None:
        """ Load model weights from the pretrained model. """
        self.model.load_weights(file_path, by_name=True)

    @staticmethod
    def distribute_model(model: Model, gpus: List[str]) -> Model:
        """ Distribute model across GPU instances. """
        try:
            parallel_model = multi_gpu_model(model, len(gpus))
            logger.info("Training using multiple GPUs..")
        except ValueError:
            parallel_model = model
            logger.info("Training using single GPU or CPU..")
        return parallel_model

    @staticmethod
    def compile_model(model: Model, optimizer: Optimizer, losses: List[Callable],
                      adversarial=False, adversarial_weight=float) -> None:
        """ The compiled model means the model configured for training. """
        char_labels = Input(name='char_labels', shape=[None], dtype='int16')
        if adversarial:
            is_synthesized = Input(name='is_synthesized', shape=[None], dtype='float')
            ctc_loss, cross_entropy = losses
            targets = {'char_probs': char_labels, 'is_synthesized_prediction': is_synthesized}
            metrics = {'char_probs': ctc_loss, 'is_synthesized_prediction': cross_entropy}
            loss_weights = [1, adversarial_weight]
            return model.compile(optimizer, losses, metrics, loss_weights, target_tensors=targets)
        else:
            targets = {'char_probs': char_labels}
            return model.compile(optimizer, losses, target_tensors=targets)

    @staticmethod
    def get_configuration(file_path: str) -> ModelConfiguration:
        """ Read components parameters from the yaml file via Configuration object. """
        return configuration.ModelConfiguration(file_path)

    @staticmethod
    def get_model(name: str, **kwargs) -> Model:
        """ Define model base on the experiment configuration. """
        if name == 'deepspeech':
            return model_module.deepspeech(**kwargs)
        elif name == 'deepspeech-custom':
            return model_module.deepspeech_custom(**kwargs)
        raise ValueError('Wrong model name')

    @staticmethod
    def get_alphabet(file_path) -> Alphabet:
        """ Alphabet consists all valid characters / phonemes and helps work with texts. """
        return text.Alphabet(file_path)

    @staticmethod
    def get_features_extractor(**kwargs) -> FeaturesExtractor:
        """ Feature Extractor helps to convert audio files to features. """
        return audio.FeaturesExtractor(**kwargs)

    @staticmethod
    def get_optimizer(name: str, **kwargs) -> Optimizer:
        """ Define optimizer - use keras documentation `keras.optimizers`. """
        if name == 'sgd':
            return SGD(**kwargs)
        elif name == 'adam':
            return Adam(**kwargs)
        raise ValueError('Wrong optimizer name')

    @staticmethod
    def get_losses(adversarial=False) -> List[Callable]:
        """ The CTC loss using TensorFlow's `ctc_loss` using Keras backend. """
        def get_length(tensor):
            lengths = tf.reduce_sum(tf.ones_like(tensor), 1)
            return tf.reshape(tf.cast(lengths, tf.int32), [-1, 1])

        def ctc_loss(y, y_hat):
            sequence_length = get_length(tf.reduce_max(y_hat, 2))
            label_length = get_length(y)
            return tf.keras.backend.ctc_batch_cost(y, y_hat, sequence_length, label_length)

        if adversarial:                              # Keras `binary_crossentropy` expects probabilities, so
            return [ctc_loss, binary_crossentropy]   # the last adversarial layer should be `sigmoid`.
        else:                                        # Keras internally reverse to logits and pass to Tensorflow.
            return [ctc_loss]

    @staticmethod
    def get_decoder(name: str, alphabet: Alphabet, model: Model, **kwargs) -> Callable:
        """ Additionally can be added the warp-ctc (GPU support). """
        if name == 'naive':
            return partial(ctc_decoder.batch_naive_decode, alphabet=alphabet, **kwargs)
        elif name == 'tensorflow':
            output_tensor, *_ = model.outputs
            decoder = ctc_decoder.get_tensorflow_decoder(output_tensor, **kwargs)
            return partial(ctc_decoder.batch_tensorflow_decode, alphabet=alphabet, decoder=decoder)
        raise ValueError('Wrong decoder name')

    @staticmethod
    def get_callbacks(home_dir: str, configurations: list, model: Model) -> List[Callback]:
        """ Define callbacks to get a view on internal states during training. """
        callbacks = []
        for configuration in configurations:
            name = configuration.pop('name')

            if name == 'TerminateOnNaN':
                callbacks.append(TerminateOnNaN())

            elif name == 'ResultKeeper':
                file_path = os.path.join(home_dir, configuration.pop('file_name'))
                callbacks.append(ResultKeeper(file_path))

            elif name == 'CustomEarlyStopping':
                callbacks.append(CustomEarlyStopping(**configuration))

            elif name == 'LearningRateScheduler':
                k = configuration.pop('k')
                lr_decay = lambda epoch, lr: lr / np.power(k, epoch)
                callbacks.append(LearningRateScheduler(lr_decay, **configuration))

            elif name == 'ReduceLROnPlateau':
                callbacks.append(ReduceLROnPlateau(**configuration))

            elif name == 'CustomModelCheckpoint':
                log_dir = os.path.join(home_dir, configuration.pop('dir_name'))
                callbacks.append(CustomModelCheckpoint(model, log_dir))

            elif name == 'CustomTensorBoard':
                log_dir = os.path.join(home_dir, configuration.pop('dir_name'))
                callbacks.append(CustomTensorBoard(log_dir))
        return callbacks
