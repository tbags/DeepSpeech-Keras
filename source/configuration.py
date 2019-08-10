import yaml
from typing import List, Dict
from source.audio import FeaturesExtractor
from source.text import Alphabet


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

    def __init__(self, file_path: str, alphabet: Alphabet, features_exractor: FeaturesExtractor):
        super().__init__(file_path)
        self.alphabet = alphabet
        self.feature_extractor = features_exractor
        self._check_file(required_keys=['constructor', 'source', 'parameters'])
        constructor_name = self._data.get('constructor')
        source = self._data.get('source')
        self.constructor = getattr(constructor_name, source)
        self.parameters = self._data.get('parameters')

    def create_generator(self):
        return self.constructor(**self.parameters)
