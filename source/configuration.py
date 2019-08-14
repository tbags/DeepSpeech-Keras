import yaml
from typing import List, Dict
from source.audio import FeaturesExtractor
from source.text import Alphabet
from source.generator import DataGenerator, AdversarialDataGenerator


class Configuration:

    def __init__(self, file_path: str):
        """ All parameters saved in .yaml file convert to dot accessible """
        self._file_path = file_path
        self._data = self._read_yaml_file()

    @property
    def data(self):
        return self._data

    def _read_yaml_file(self) -> Dict:
        """ Read YAML configuration file """
        with open(self._file_path, 'r') as stream:
            return yaml.load(stream)

    def _check_file(self, required_keys: List[str]):
        if not all(key in self._data for key in required_keys):
            raise KeyError(f'Configuration file should have all required keys: {required_keys}')


class ModelConfiguration(Configuration):
    """
    Each DeepSpeech model has own model.yaml configuration file. This
    configuration object passes through all methods required to build
    DeepSpeech object.
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._check_file(required_keys=['features_extractor', 'model', 'callbacks', 'optimizer', 'decoder'])
        self.features_extractor = self._data.get('features_extractor')
        self.model = self._data.get('model')
        self.callbacks = self._data.get('callbacks')
        self.optimizer = self._data.get('optimizer')
        self.decoder = self._data.get('decoder')


class DatasetConfiguration(Configuration):

    def __init__(self, file_path: str, alphabet: Alphabet, features_extractor: FeaturesExtractor):
        super().__init__(file_path)
        self._check_file(required_keys=['class_name', 'source', 'parameters'])
        self.alphabet = alphabet
        self.features_extractor = features_extractor
        self.class_name = self._data.get('class_name')
        self.source = self._data.get('source')
        self.parameters = self._data.get('parameters')

    def create_generator(self):
        dependencies = dict(
            alphabet=self.alphabet,
            features_extractor=self.features_extractor,
        )
        if self.source == 'audio_files':
            class_method = 'from_audio_files'
        elif self.source == 'prepared_features':
            class_method = 'from_prepared_features'
        else:
            raise ValueError(f'Wrong defined source: {self.source}')
        if self.class_name == 'DataGenerator':
            return getattr(DataGenerator, class_method)(**dependencies, **self.parameters)
        elif self.class_name == 'AdversarialDataGenerator':
            return getattr(AdversarialDataGenerator, class_method)(
                [{**dependencies, **kwargs} for kwargs in self.parameters]
            )
        else:
            raise ValueError(f'Wrong defined class_name: {self.class_name}')
