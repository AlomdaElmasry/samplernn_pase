import numpy as np
import yaml
from .utils import SampleRNNLogger


class SampleRNNConfiguration:
    """
    Configuration handler for the SampleRNN architecture
    """

    logger: SampleRNNLogger

    datasets = {
        'vctk': {
            'speakers-info_file_path': None,
            'wavs_folder_path': None,
            'conds_utterance': {
                'acoustic_folder_path': None,
                'linguistic_folder_path': None
            },
            'sampling_freq': None,
            'n_speakers': None,
            'enabled': True
        },
        'cmu_artic': {
            'info_file_path': None,
            'wavs_folder_path': None,
            'conds_utterance': {
                'acoustic_folder_path': None,
                'linguistic_folder_path': None
            },
            'sampling_freq': None,
            'n_speakers': None,
            'enabled': True
        },
        'total_speakers': None
    }

    pase = {
        'config_file_path': None,
        'trained_model_path': None
    }

    split = {
        'modeling_male_speakers': None,
        'modeling_female_speakers': None,
        'modeling_train_time_per_speaker': None,
        'modeling_val_time_per_speaker': None,
        'modeling_test_time_per_speaker': None,
        'adaptation_male_speakers': None,
        'adaptation_female_speakers': None,
        'adaptation_train_time_per_speaker': None,
        'adaptation_val_time_per_speaker': None,
        'adaptation_test_time_per_speaker': None,
        'min_utterance_samples': 0,
        'max_utterance_samples': 999999,
        'priorize_longer_speakers': True,
        'priorize_longer_utterances': False
    }

    quantizer = {
        'q_type_ulaw': True,
        'q_levels': 256
    }

    conditionants = {
        'speaker_type': 'embedding',
        'speaker_size': 0,
        'speaker_embedding_size': 0,
        'speaker_pase_seed_size': 0,
        'utterance_type': None,
        'utterance_size': 0,
        'utterance_size_expanded': 0,
        'utterance_acoustic_size': 0,
        'utterance_linguistic_size': 0,
        'utterance_linguistic_phonemes_embedding_size': 0,
        'utterance_linguistic_vowels_embedding_size': 0,
        'utterance_linguistic_gpos_embedding_size': 0,
        'utterance_linguistic_tobi_embedding_size': 0,
        'global_size': 0
    }

    architecture = {
        'sequence_length': 13,
        'frame_layers_ratios': [20, 20],
        'frame_layers_fs': None,
        'frame_layers_rnn_layers': [1, 1],
        'frame_layers_rnn_hidden_size': [1024, 1024],
        'frame_size': None,
        'receptive_field': None
    }

    training = {
        'batch_size': 256,
        'max_epochs': 100,
        'lr': 1e-3,
        'lr_scheduler': True,
        'lr_scheduler_patience': 3,
        'lr_scheduler_factor': 0.5
    }

    adaptation = {
        'freeze_model': True,
        'freeze_embedding': True
    }

    def __init__(self, logger: SampleRNNLogger):
        self.logger = logger

    def load_file(self, config_file_path: str, initialize_computed_parameters: bool = False):
        with open(config_file_path, 'r') as config_file:
            config_content = yaml.load(config_file, Loader=yaml.FullLoader)
            for conf_key, conf_val in config_content.items():
                if hasattr(self, conf_key):
                    setattr(self, conf_key, conf_val)
                else:
                    self.logger.error('The configuration parameter "{}" is not valid'.format(conf_key))
                    exit()

        # Initialize the computer parameters
        if initialize_computed_parameters:
            self._validate_config()
            self._initialize_computed_parameters()

    def _initialize_computed_parameters(self):

        # Initialize self.datasets['total_speakers'] parameter
        self.datasets['total_speakers'] = 0
        if self.datasets['vctk']['enabled']:
            self.datasets['total_speakers'] += self.datasets['vctk']['n_speakers']
        if self.datasets['cmu_artic']['enabled']:
            self.datasets['total_speakers'] += self.datasets['cmu_artic']['n_speakers']

        # Intialize self.conditionants['speaker_size']
        if self.conditionants['speaker_type'] == 'embedding':
            self.conditionants['speaker_size'] = self.conditionants['speaker_embedding_size']
        elif self.conditionants['speaker_type'] == 'embedding_pase_init':
            self.conditionants['speaker_size'] = self.conditionants['speaker_embedding_size']
        else:
            self.conditionants['speaker_size'] = self.conditionants['speaker_pase_seed_size']

        # Initialize self.conditionants['utterance_size'] and conditionants['utterance_size_expanded'] parameters
        if self.conditionants['utterance_type'] == 'acoustic':
            self.conditionants['utterance_size'] = self.conditionants['utterance_size_expanded'] = self.conditionants[
                'utterance_acoustic_size']
        else:
            self.conditionants['utterance_size'] = self.conditionants['utterance_linguistic_size']
            self.conditionants['utterance_size_expanded'] = \
                self.conditionants['utterance_linguistic_size'] - 10 + \
                5 * self.conditionants['utterance_linguistic_phonemes_embedding_size'] + \
                1 * self.conditionants['utterance_linguistic_vowels_embedding_size'] + \
                3 * self.conditionants['utterance_linguistic_gpos_embedding_size'] + \
                1 * self.conditionants['utterance_linguistic_tobi_embedding_size']
        if self.conditionants['utterance_type'] == 'linguistic_lf0':
            self.conditionants['utterance_size'] += 2
            self.conditionants['utterance_size_expanded'] += 2

        # Intialize self.architecture['frame_size'] parameter
        self.architecture['frame_size'] = np.prod(self.architecture['frame_layers_ratios'])

        # Intialize self.architecture['frame_layers_fs'] parameter
        self.architecture['frame_layers_fs'] = list(map(int, np.cumprod(self.architecture['frame_layers_ratios'])))

        # Intialize self.architecture['receptive_field'] parameter
        self.architecture['receptive_field'] = self.architecture['frame_size'] * self.architecture['sequence_length']

    def _validate_config(self):
        assert self.conditionants['speaker_type'] in ['embedding', 'pase_seed', 'pase_trained']
        assert self.conditionants['utterance_type'] in ['acoustic', 'linguistic', 'linguistic_lf0']