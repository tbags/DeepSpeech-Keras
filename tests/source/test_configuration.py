import mock
from source.configuration import ModelConfiguration, DatasetConfiguration
from source.utils import chdir

chdir(to='ROOT')


def test_model_configuration():
    alphabet = mock.MagicMock()
    features_extractor = mock.MagicMock()
    config = DatasetConfiguration('tests/data/train-dataset.yaml', alphabet, features_extractor)
    generator = config.create_generator()
    assert len(generator) == 4
    part_1, part_2 = generator._generators
    assert isinstance(part_1.mask_params, dict)
    assert not part_2.mask_params
    assert generator._generator_sizes == [2, 2]
    assert part_1.is_adversarial and part_1.is_synthesized
    assert part_2.is_adversarial and not part_2.is_synthesized
    config = DatasetConfiguration('tests/data/dev-dataset.yaml', alphabet, features_extractor)
    generator = config.create_generator()
    assert not generator.mask_params


def test_dataset_configuration():
    config = ModelConfiguration('tests/models/base/configuration.yaml')
    assert isinstance(config.model, dict)
    assert isinstance(config.callbacks, list)
    assert isinstance(config.optimizer, dict)
    assert isinstance(config.decoder, dict)
