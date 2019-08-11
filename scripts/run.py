import os
import argparse

from source.configuration import DatasetConfiguration
from source.deepspeech import DeepSpeech
from source.utils import chdir, create_logger


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--home_dir', required=True, help='DeepSpeech home directory')
    parser.add_argument('--alphabet', help='The Alphabet file. The deafault is `{home_dir}/alphabet.txt`')
    parser.add_argument('--configuration', help='Model configuration file. The deafault is `{home_dir}/configuration.yaml`')
    parser.add_argument('--train', help='Train Dataset configuration file. The deafault is `{home_dir}/train-dataset.yaml`')
    parser.add_argument('--dev', help='Dev Dataset configuration file. The deafault is `{home_dir}/dev-dataset.yaml`')
    parser.add_argument('--weights', help='Destination of the trained model weights. The deafault is `{home_dir}/weights.hdf5`')
    parser.add_argument('--pretrained_weights', help='Use this weights from the pretrained model')
    parser.add_argument('--log', help='The log file. The deafault is `{home_dir}/trainig.log`')
    parser.add_argument('--log_level', type=int, default=20, help='Log level')
    parser.add_argument('--epochs', type=int, default=25, help='Epochs during training')
    return parser


def main(
        model_config_path: str,
        alphabet_path: str,
        train_path: str,
        dev_path: str,
        epochs: int,
        pretrained_weights: str = ''
):
    deepspeech = DeepSpeech.construct(model_config_path, alphabet_path)
    if pretrained_weights:
        deepspeech.load(pretrained_weights)

    dependencies = deepspeech.alphabet, deepspeech.features_extractor
    train_generator = DatasetConfiguration(train_path, *dependencies).create_generator()
    dev_generator = DatasetConfiguration(dev_path, *dependencies).create_generator()
    deepspeech.fit(train_generator, dev_generator, epochs=epochs, shuffle=False)
    deepspeech.save(WEIGHTS_PATH)


if __name__ == "__main__":
    chdir(to='ROOT')
    PARSER = create_parser()

    ARGS = PARSER.parse_args()
    home = lambda name: os.path.join(ARGS.home_dir, name)
    ALPHABET_PATH = ARGS.alphabet if ARGS.alphabet else home('alphabet.txt')
    MODEL_CONFIG_PATH = ARGS.home_dir if ARGS.home_dir else home('configuration.yaml')
    TRAIN_PATH = ARGS.train if ARGS.train else home('train-dataset.yaml')
    DEV_PATH = ARGS.dev if ARGS.dev else home('dev-dataset.yaml')
    WEIGHTS_PATH = ARGS.weights if ARGS.weights else home('weights.hdf5')
    LOG_PATH = ARGS.log if ARGS.log else home('training.log')

    logger = create_logger(LOG_PATH, level=ARGS.log_level, name='deepspeech')
    logger.info(f'Arguments: \n{ARGS}')

    main(
        model_config_path=MODEL_CONFIG_PATH,
        alphabet_path=ALPHABET_PATH,
        train_path=TRAIN_PATH,
        dev_path=DEV_PATH,
        epochs=ARGS.epoch,
        pretrained_weights=ARGS.pretrained_weights
    )
