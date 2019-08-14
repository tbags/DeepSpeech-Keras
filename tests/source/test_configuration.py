import mock
from source.configuration import ModelConfiguration, DatasetConfiguration
from source.utils import chdir

chdir(to='ROOT')


def test_dataset_configuration():
    config = ModelConfiguration('tests/models/base/configuration.yaml')
    assert isinstance(config.model, dict)
    assert isinstance(config.callbacks, list)
    assert isinstance(config.optimizer, dict)
    assert isinstance(config.decoder, dict)
