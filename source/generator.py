from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from keras.utils import Sequence
from source.text import Alphabet
from source.audio import FeaturesExtractor
from source.augmentation import mask_features


class DataGenerator(Sequence):
    """
    Generates data for Keras

    `Sequence` are a safer way to do multiprocessing. This structure
    guarantees that the network will only train once on each sample per epoch
    which is not the case with generators.

    References:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    """

    def __init__(self,
                 references: pd.DataFrame,
                 alphabet: Alphabet,
                 features_extractor: FeaturesExtractor,
                 shuffle_after_epoch: int = 1,
                 batch_size: int = 30,
                 features_store: h5py.File = None,
                 mask: bool = False,
                 mask_params: Dict[str, Any] = None,
                 is_adversarial: bool = False,
                 is_synthesized: bool = False):
        self._references = references
        self._features_store = features_store
        self._features_extractor = features_extractor
        self._batch_size = batch_size
        self.alphabet = alphabet
        self.shuffle_after_epoch = shuffle_after_epoch
        self.is_adversarial = is_adversarial
        self.is_synthesized = is_synthesized
        self.epoch = 0
        self.indices = np.arange(len(self))
        self.mask = mask
        self.mask_params = mask_params

    @classmethod
    def from_audio_files(cls, file_path, **kwargs) -> "DataGenerator":
        """ Create generator from csv file. The file contains audio file paths
        with corresponding transcriptions. """
        references = pd.read_csv(file_path, usecols=['path', 'transcript'], sep=',', encoding='utf-8', header=0)
        return cls(references, **kwargs)

    @classmethod
    def from_prepared_features(cls, file_path, **kwargs) -> "DataGenerator":
        """ Create generator from prepared features saved in the HDF5 format.
        The hdf5 file has the hierarchy with /-separator and also can be invoke via `path`. """
        features_store = h5py.File(file_path, mode='r')
        references = pd.HDFStore(file_path, mode='r')['references']  # Read DataFrame via PyTables
        return cls(references, features_store=features_store, **kwargs)

    def __len__(self) -> int:
        """ Denotes the number of batches per epoch. """
        return int(np.floor(len(self._references.index) / self._batch_size))

    def __getitem__(self, index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """ Operator to get the batch data. """
        batch_index = self.indices[index]
        return self._get_batch(batch_index)

    def _get_batch(self, index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """ Read (if features store exist) or generate features and labels batch. """
        start, end = index * self._batch_size, (index + 1) * self._batch_size
        references = self._references[start:end]
        paths, transcripts = references.path, references.transcript

        labels = self.alphabet.get_batch_labels(transcripts)
        if self._features_store:
            features = self._read_features(paths)
        else:
            features = self._extract_features(paths)
        if self.mask:
            features = self._mask_features(features)
        if self.is_adversarial:
            is_synthesized_labels = np.zeros(self._batch_size) + self.is_synthesized
            return {'X': features}, {'main_output': labels, 'is_synthesized': is_synthesized_labels}
        else:
            return {'X': features}, {'main_output': labels}

    def _read_features(self, paths: List[str]) -> np.ndarray:
        """ Read already prepared features from the store. """
        features = [self._features_store[path][:] for path in paths]
        return self._features_extractor.align(features)

    def _extract_features(self, paths) -> np.ndarray:
        """ Extract features from the audio files (mono 16kHz). """
        return self._features_extractor.get_features(files=paths)

    def _mask_features(self, features: np.ndarray) -> np.ndarray:
        """ SpecAugment: A Simple Data Augmentation Method. """
        return np.stack([mask_features(sample, **self.mask_params) for sample in features], axis=0)

    def on_epoch_end(self) -> None:
        """ Invoke methods at the end of the each epoch. The fit method should have: `shuffle=False`.
        Keras OrderedEnqueuer seems to run on async on two threads so the epoch number is counted twice (bug). """
        self.epoch += 1
        self._shuffle_indices()

    def _shuffle_indices(self) -> None:
        """ Set up the order of next batches """
        if self.epoch >= self.shuffle_after_epoch:
            np.random.shuffle(self.indices)


class AdversarialDataGenerator(Sequence):

    def __init__(self, generators: List[DataGenerator]):
        self._generator, *_ = [g for g in generators if not g.is_synthesized]       # Takes only the first one
        self._synthesized_generators = [g for g in generators if g.is_synthesized]
        self._synthesized_generators_sizes = [len(generator) for generator in self._synthesized_generators]
        self._synthesized_generators_limits = np.cumsum(self._synthesized_generators_sizes)
        self.epoch = 0
        self.indices = np.arange(len(self))
        np.random.shuffle(self.indices)

    @classmethod
    def from_audio_files(cls, generators_params: List[Dict]) -> "AdversarialDataGenerator":
        generators = []
        for generator_parameters in generators_params:
            generator = DataGenerator.from_audio_files(**generator_parameters)
            generators.append(generator)
        return cls(generators)

    @classmethod
    def from_prepared_features(cls, generators_params: List[Dict]) -> "AdversarialDataGenerator":
        generators = []
        for generator_parameters in generators_params:
            generator = DataGenerator.from_prepared_features(**generator_parameters)
            generators.append(generator)
        return cls(generators)

    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return len(self._generator)

    def __getitem__(self, next_index):
        """ Operator to get the batch data. """
        index = self.indices[next_index]
        rich_samples = self._generator[index]
        synthesized_samples = self.get_synthesized_samples(index)
        batch = self.concatenate(rich_samples, synthesized_samples)
        return batch

    def concatenate(self, batch, batch_syn):
        def concat(rich: np.ndarray, synthesized: np.ndarray):
            new = np.zeros_like(np.concatenate([rich, synthesized]))
            new[rich_index] = rich
            new[syn_index] = synthesized
            return new

        def concat_features(rich: np.ndarray, synthesized: np.ndarray):
            aligned = FeaturesExtractor.align([*rich, *synthesized])
            rich = aligned[:len(rich)]
            synthesized = aligned[len(rich):]
            new = np.zeros_like(aligned)
            new[rich_index] = rich
            new[syn_index] = synthesized
            return new

        def concat_transcripts(rich: np.ndarray, synthesized: np.ndarray):
            arrays = [*rich, *synthesized]
            max_array = max(arrays, key=len)
            default = self._generator.alphabet.blank_token
            new = np.full(shape=[len(arrays), *max_array.shape], fill_value=default)
            for i, array in enumerate(arrays):
                size, = array.shape
                new[i, :size] = array
            return new

        (inputs, targets), (inputs_syn, targets_syn) = batch, batch_syn
        X, X_syn = inputs['X'], inputs_syn['X']
        y, y_syn = targets['main_output'], targets_syn['main_output']
        labels, labels_syn = targets['is_synthesized'], targets_syn['is_synthesized']

        index = np.arange(len(labels)+len(labels_syn))
        syn_index = np.random.choice(index, len(labels_syn), replace=False)
        rich_index = list(set(range(len(index))) - set(syn_index))
        return {'X': concat_features(X, X_syn)}, \
               {'main_output': concat_transcripts(y, y_syn), 'is_synthesized': concat(labels, labels_syn)}

    def get_synthesized_samples(self, index):
        generator_index = np.searchsorted(self._synthesized_generators_limits, index, side='right')     # Right side because limits are based on the lengths, not indices
        generator = self._synthesized_generators[generator_index]
        if generator_index == 0:
            gen_index = index
        else:
            gen_index = index - self._synthesized_generators_limits[generator_index-1]
        return generator[gen_index]

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        self.epoch += 1
        self._generator.on_epoch_end()
        for generator in self._synthesized_generators:      # Do even if all generators are not exhaust.
            generator.on_epoch_end()
