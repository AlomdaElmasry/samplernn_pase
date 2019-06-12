import ahoproc_tools.io
import ahoproc_tools.interpolate
import concurrent.futures
import soundfile
import csv
import numpy as np
import os
import pickle
import random
import operator
from .configuration import SampleRNNConfiguration
from .utils import SampleRNNLogger, SampleRNNLabReader


class SampleRNNData:
    """

    """

    datasets_info = {}
    speakers_info = {}
    utterances_info = {}
    utterances_conds_linguistic_categories = {
        'phonemes': set(),
        'vowels': set(),
        'gpos': set(),
        'tobi': set()
    }

    modeling_speakers_ids = []
    modeling_utterances_ids_train = []
    modeling_utterances_ids_val = []
    modeling_utterances_ids_test = []

    adaptation_speakers_ids = []
    adaptation_utterances_ids_train = []
    adaptation_utterances_ids_val = []
    adaptation_utterances_ids_test = []

    conf: SampleRNNConfiguration
    logger: SampleRNNLogger
    thread_pool: concurrent.futures.ThreadPoolExecutor

    def __init__(self, conf: SampleRNNConfiguration, logger: SampleRNNLogger):
        self.conf = conf
        self.logger = logger
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=40)

    def create_data_main(self):
        self._load_datasets_info()
        with self.thread_pool as executor:
            for dataset_id, _ in self.datasets_info.items():
                self._load_speakers_info(dataset_id)
            for speaker_id, _ in self.speakers_info.items():
                executor.submit(self._load_utterances_info, speaker_id)
                executor.submit(self._load_conds_acoustic_stads, speaker_id)
                executor.submit(self._load_conds_linguistic_stads, speaker_id)

    def create_data_split(self):
        self._allocate_speakers_ids()
        self._allocate_utterances_ids()

    def save_data_main(self, data_main_file_path: str):
        with open(data_main_file_path, 'wb') as data_main_file:
            pickle.dump((self.speakers_info, self.utterances_info,
                         self.utterances_conds_linguistic_categories), data_main_file)

    def save_data_split(self, data_split_file_path: str):
        with open(data_split_file_path, 'wb') as data_split_file:
            pickle.dump((self.modeling_speakers_ids, self.modeling_utterances_ids_train,
                         self.modeling_utterances_ids_val, self.modeling_utterances_ids_test,
                         self.adaptation_speakers_ids, self.adaptation_utterances_ids_train,
                         self.adaptation_utterances_ids_val, self.adaptation_utterances_ids_test), data_split_file)

    def load_data_main(self, data_main_file_path: str):
        self._load_datasets_info()
        with open(data_main_file_path, 'rb') as data_main_file:
            self.speakers_info, self.utterances_info, self.utterances_conds_linguistic_categories \
                = pickle.load(data_main_file)

    def load_data_split(self, data_split_file_path: str):
        with open(data_split_file_path, 'rb') as data_split_file:
            self.modeling_speakers_ids, self.modeling_utterances_ids_train, self.modeling_utterances_ids_val, \
            self.modeling_utterances_ids_test, self.adaptation_speakers_ids, self.adaptation_utterances_ids_train, \
            self.adaptation_utterances_ids_val, self.adaptation_utterances_ids_test = pickle.load(data_split_file)

    @staticmethod
    def get_utterance_id(utterance_info):
        return utterance_info['speaker']['name'] + '-' + utterance_info['utterance']['name']

    def _read_dataset_info_file(self, dataset_id):

        # Create an empty list to store the info
        info_list = []

        # Open the INFO file
        with open(self.datasets_info[dataset_id]['info_file_path'], 'r') as info_file:

            # Handle the VCTK info file
            if dataset_id == 'vctk':
                speakers_info = csv.reader(info_file, delimiter=' ')
                speakers_info_it = iter(speakers_info)
                next(speakers_info_it)
                for speakers_info_row in speakers_info_it:
                    info_list.append(('p' + speakers_info_row[0], speakers_info_row[4]))

            # Handle the CMU Artic info file
            if dataset_id == 'cmu_artic':
                speakers_info = csv.reader(info_file, delimiter=',')
                speakers_info_it = iter(speakers_info)
                next(speakers_info_it)
                for speakers_info_row in speakers_info_it:
                    info_list.append((speakers_info_row[0], speakers_info_row[1]))

        # Return the list of tuples
        return info_list

    def _load_datasets_info(self):

        # Create an entry for the VCTK dataset
        if 'vctk' in self.conf.datasets:
            self.datasets_info['vctk'] = {
                'index': len(self.datasets_info),
                'name': 'vctk',
                'speakers_prefix': 'vctk_',
                'info_file_path': self.conf.datasets['vctk']['info_file_path'],
                'wavs_folder_path': self.conf.datasets['vctk']['wavs_folder_path'],
                'conds_utterance': {
                    'acoustic_folder_path': self.conf.datasets['vctk']['conds_utterance']['acoustic_folder_path'],
                    'linguistic_folder_path': self.conf.datasets['vctk']['conds_utterance']['linguistic_folder_path'],
                },
                'speakers_count': self.conf.datasets['vctk']['n_speakers'],
                'utterances_sf': self.conf.datasets['vctk']['sampling_freq']
            }

        # Create an entry for the CMU Artic dataset
        if 'cmu_artic' in self.conf.datasets:
            self.datasets_info['cmu_artic'] = {
                'index': len(self.datasets_info),
                'name': 'cmu_artic',
                'speakers_prefix': 'cmu_',
                'info_file_path': self.conf.datasets['cmu_artic']['info_file_path'],
                'wavs_folder_path': self.conf.datasets['cmu_artic']['wavs_folder_path'],
                'conds_utterance': {
                    'acoustic_folder_path': self.conf.datasets['cmu_artic']['conds_utterance']['acoustic_folder_path'],
                    'linguistic_folder_path': self.conf.datasets['cmu_artic']['conds_utterance'][
                        'linguistic_folder_path'],
                },
                'speakers_count': self.conf.datasets['cmu_artic']['n_speakers'],
                'utterances_sf': self.conf.datasets['cmu_artic']['sampling_freq'],

            }

    def _load_speakers_info(self, dataset_id):
        """

        Returns:

        """

        # Get the dataset information
        dataset = self.datasets_info[dataset_id]

        # Get a formated list of (speaker_id, gender)
        info_list = self._read_dataset_info_file(dataset_id)

        # Iterate over the list
        for speaker_name, speaker_gender in info_list:
            self.speakers_info[dataset['speakers_prefix'] + speaker_name] = {
                'index': len(self.speakers_info),
                'dataset_id': dataset_id,
                'name': speaker_name,
                'gender': speaker_gender,
                'wavs_len': 0,
                'conds_acoustic_stads': (0, 0),
                'conds_linguistic_stads': (0, 0)
            }

    def _load_utterances_info(self, speaker_id):
        """

        Args:
            speaker_id (str):
        Returns:

        """

        # Get speaker info
        speaker = self.speakers_info[speaker_id]

        # Get dataset info
        dataset = self.datasets_info[speaker['dataset_id']]

        # Define PATH to look into
        speakers_wavs_folder_path = dataset['wavs_folder_path'] + speaker['name'] + os.sep

        # Iterate over each FILE of the current speaker
        for _, _, utterances_names in os.walk(speakers_wavs_folder_path):

            # Iterate over all the files
            for utterance_name in utterances_names:

                # Verify that the current file is a .lab and not any trash file
                if utterance_name[-4:] != '.wav':
                    continue

                # Get utterance id
                utterance_id = speaker_id + '-' + utterance_name[:-4]

                # Load the utterance
                utterance, _ = soundfile.read(speakers_wavs_folder_path + utterance_name)

                # Store important data in the dict of the utterance
                self.utterances_info[utterance_id] = {
                    'index': len(self.utterances_info),
                    'speaker_id': speaker_id,
                    'name': utterance_name[:-4],
                    'wav_len': utterance.size / dataset['utterances_sf'],
                    'path': speaker['name'] + os.sep + utterance_name[:-4]
                }

                # Add duration to speaker
                speaker['wavs_len'] += self.utterances_info[utterance_id]['wav_len']

    def _load_conds_acoustic_stads(self, speaker_id):
        """

        Args:
            speaker:

        Returns:

        """

        # Get speaker dict
        speaker = self.speakers_info[speaker_id]

        # Get dataset info
        dataset = self.datasets_info[speaker['dataset_id']]

        # Define PATH to look into
        speaker_conds_acoustic_dir = dataset['conds_acoustic_folder_path'] + speaker['name'] + os.sep

        # Auxiliar variable to store the REAL features and compute MEAN and STD
        utterances_aux = None

        # Iterate over each FILE of the current speaker
        for _, _, utterances_names in os.walk(speaker_conds_acoustic_dir):

            # Iterate over all the files
            for utterance_name in utterances_names:

                # Verify that the current file is a .lab and not any trash file
                if utterance_name[-3:] != '.cc':
                    continue

                # Load Acoustic conds
                utterance_cc = ahoproc_tools.io \
                    .read_aco_file(speaker_conds_acoustic_dir + utterance_name, (-1, 40))
                utterance_fv = ahoproc_tools.io \
                    .read_aco_file(speaker_conds_acoustic_dir + utterance_name[:-3] + '.fv', (-1))
                utterance_lf0 = ahoproc_tools.io \
                    .read_aco_file(speaker_conds_acoustic_dir + utterance_name[:-3] + '.lf0', (-1))

                # Interpolate both FV and LF0 and obtain VU
                utterance_fv, _ = ahoproc_tools.interpolate.interpolation(utterance_fv, 1e3)
                utterance_lf0, utterance_vu = ahoproc_tools.interpolate.interpolation(utterance_lf0, -1e10)

                # Compute LOG(FV)
                utterance_fv = np.log(utterance_fv)

                # Merge all the conds, set 1 in the last position so we get MEAN=0 & STD=0 for VU bool conditionant
                utterance_conds = np.concatenate([
                    utterance_cc,
                    np.expand_dims(utterance_fv, 1),
                    np.expand_dims(utterance_lf0, 1),
                    np.zeros((utterance_cc.shape[0], 1))
                ], axis=1)

                # Store FEATURES to obtain the MEAN and STD
                if utterances_aux is None:
                    utterances_aux = utterance_conds
                else:
                    utterances_aux = np.concatenate([utterances_aux, utterance_conds])

        # Compute the MEAN and STD
        utterances_means = np.mean(utterances_aux, axis=0)
        utterances_stds = np.std(utterances_aux, axis=0)

        # Set the STD=1 for the bool conditionant
        utterances_stds[-1] = 1

        # Store MEAN and STD in the information of the speaker
        speaker['conds_acoustic_stads'] = (utterances_means.tolist(), utterances_stds.tolist())

    def _load_conds_linguistic_stads(self, speaker_id):
        """

        Args:
            speaker_id (str):

        Returns:

        """

        # Get speaker dict
        speaker = self.speakers_info[speaker_id]

        # Get dataset info
        dataset = self.datasets_info[speaker['dataset_id']]

        # Define PATH to look into
        speaker_conds_linguistic_dir = dataset['conds_linguistic_folder_path'] + speaker['name'] + os.sep

        # Auxiliar variable to store the REAL features and compute MEAN and STD
        conds_linguistic_aux = []

        # Iterate over each FILE of the current speaker
        for _, _, conds_linguistic_names in os.walk(speaker_conds_linguistic_dir):

            # Iterate over all the files
            for conds_linguistic_name in conds_linguistic_names:

                # Verify that the current file is a .lab and not any trash file
                if conds_linguistic_name[-4:] != '.lab':
                    continue

                # Iterate over each LINE of the current .lab file
                for lab_line in \
                        SampleRNNLabReader.read_lab(lab_file_path=speaker_conds_linguistic_dir + conds_linguistic_name):
                    # Store the PHONEMES features in the class' set
                    # PHONEMES Features: p1, p2, p3, p4, p5
                    [self.utterances_conds_linguistic_categories['phonemes'].add(i) for i in operator
                        .itemgetter(2, 3, 4, 5, 6)(lab_line)]

                    # Store the Vowel feature in the class' set
                    # Vowel Features: b16
                    self.utterances_conds_linguistic_categories['vowels'].add(lab_line[27])

                    # Store the GPOS features in the class' set
                    # GPOS Features: d1, e1, f1
                    [self.utterances_conds_linguistic_categories['gpos'].add(i) for i in operator
                        .itemgetter(31, 33, 41)(lab_line)]

                    # Store the TOBI feature in the class' set
                    # TOBI Features: h5
                    self.utterances_conds_linguistic_categories['tobi'].add(lab_line[49])

                    # Store the duration to obtain the MEAN and STD
                    conds_linguistic_aux.append(int(lab_line[1]) - int(lab_line[0]))

                    # Store the Real features to obtain the MEAN and STD
                    # Real Features: p6, p7, a3, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, c3, d2, e2,
                    # e3, e4, e5, e6, e7, e8, f2, g1, g2, h1, h2, h3, h4, i1, i2, j1, j2, j3
                    conds_linguistic_aux += list(operator.itemgetter(
                        7, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 32, 34, 35, 36, 37, 38, 39,
                        40,
                        42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54
                    )(lab_line))

        # Reshape and fix unknown values of the auxiliar variable to perform MEAN and STD
        utterances_aux_np = np.asarray(conds_linguistic_aux).reshape(-1, 38)
        utterances_aux_np[utterances_aux_np == 'x'] = 0
        utterances_aux_np = utterances_aux_np.astype(np.float)

        # Create vector of MEANs and STDs
        utterances_means = np.zeros(55, )
        utterances_stds = np.ones(55, )

        # Assign the values to the correct positions
        np.put(
            a=utterances_means,
            ind=[0, 7, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 32, 34, 35, 36, 37, 38, 39, 40,
                 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54],
            v=np.mean(utterances_aux_np, axis=0)
        )

        # Assign the values to the correct positions
        np.put(
            a=utterances_stds,
            ind=[0, 7, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 32, 34, 35, 36, 37, 38, 39, 40,
                 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54],
            v=np.std(utterances_aux_np, axis=0)
        )

        # Store MEAN and STD in the speaker
        speaker['conds_linguistic_stads'] = (
            utterances_means.tolist(),
            utterances_stds.tolist()
        )

    def _allocate_speakers_ids(self):
        # Filter male speakers
        male_speakers_ids = [
            speaker_id for speaker_id, speaker in self.speakers_info.items() if speaker['gender'] == 'M'
        ]

        # Filter female speakers
        female_speakers_ids = [
            speaker_id for speaker_id, speaker in self.speakers_info.items() if speaker['gender'] == 'F'
        ]

        # Verify that there are enough male speakers
        if len(male_speakers_ids) < (
                self.conf.split['modeling_male_speakers'] +
                self.conf.split['adaptation_male_speakers']
        ):
            self.logger.error('There are not enough male speakers')
            exit()

        # Verify that there are enough female speakers
        if len(female_speakers_ids) < (
                self.conf.split['modeling_female_speakers'] +
                self.conf.split['adaptation_female_speakers']
        ):
            self.logger.error('There are not enough female speakers')
            exit()

        # By default, order the lists by DESC duration of the speakers wavs
        if self.conf.split['priorize_longer_speakers']:
            male_speakers_ids = sorted(
                male_speakers_ids,
                key=(lambda speaker_id: self.speakers_info[speaker_id]['wavs_len']),
                reverse=True
            )
            female_speakers_ids = sorted(
                female_speakers_ids,
                key=(lambda speaker_id: self.speakers_info[speaker_id]['wavs_len']),
                reverse=True
            )

        # If not, just shuffle the set
        else:
            random.shuffle(male_speakers_ids)
            random.shuffle(female_speakers_ids)

        # Get the speakers
        modeling_male_speakers_ids = male_speakers_ids[:self.conf.split['modeling_male_speakers']]
        modeling_female_speakers_ids = female_speakers_ids[:self.conf.split['modeling_female_speakers']]
        adaptation_male_speakers_ids = male_speakers_ids[
                                       self.conf.split['modeling_male_speakers']:
                                       (self.conf.split['modeling_male_speakers'] + self.conf.split[
                                           'adaptation_male_speakers'])
                                       ]
        adaptation_female_speakers_ids = female_speakers_ids[
                                         self.conf.split['modeling_female_speakers']:
                                         (self.conf.split['modeling_female_speakers'] + self.conf.split[
                                             'adaptation_female_speakers'])
                                         ]

        # Assign the speakers to the lists
        self.modeling_speakers_ids = modeling_male_speakers_ids + modeling_female_speakers_ids
        self.adaptation_speakers_ids = adaptation_male_speakers_ids + adaptation_female_speakers_ids

        # Sort the speakers ids to the natural order
        self.modeling_speakers_ids.sort()
        self.adaptation_speakers_ids.sort()

    def _allocate_utterances_ids(self, shuffle=True):
        # Iterate over modeling speakers ids
        for modeling_speaker_id in self.modeling_speakers_ids:
            # Get the list of utterances ids of the speaker
            modeling_speaker_utterances_ids = [
                utterance_id for utterance_id, utterance in self.utterances_info.items()
                if utterance['speaker_id'] == modeling_speaker_id
            ]

            # Shuffle the utterances, if required
            if shuffle:
                random.shuffle(modeling_speaker_utterances_ids)

            # Fill the test split of modeling
            modeling_speaker_test_time_acc = 0
            while modeling_speaker_test_time_acc < self.conf.split['modeling_test_time_per_speaker']:
                if len(modeling_speaker_utterances_ids) > 0:
                    modeling_speaker_utterance_id = modeling_speaker_utterances_ids.pop()
                    self.modeling_utterances_ids_test.append(modeling_speaker_utterance_id)
                    modeling_speaker_test_time_acc += self.utterances_info[modeling_speaker_utterance_id]['wav_len']
                else:
                    self.logger.warning('Not enough data for {}'.format(modeling_speaker_id))
                    break

            # Fill the validation split of modeling
            modeling_speaker_val_time_acc = 0
            while modeling_speaker_val_time_acc < self.conf.split['modeling_val_time_per_speaker']:
                if len(modeling_speaker_utterances_ids) > 0:
                    modeling_speaker_utterance_id = modeling_speaker_utterances_ids.pop()
                    self.modeling_utterances_ids_val.append(modeling_speaker_utterance_id)
                    modeling_speaker_val_time_acc += self.utterances_info[modeling_speaker_utterance_id]['wav_len']
                else:
                    self.logger.warning('Not enough data for {}'.format(modeling_speaker_id))
                    break

            # Fill the train split of modeling
            modeling_speaker_train_time_acc = 0
            while modeling_speaker_train_time_acc < self.conf.split['modeling_train_time_per_speaker']:
                if len(modeling_speaker_utterances_ids) > 0:
                    modeling_speaker_utterance_id = modeling_speaker_utterances_ids.pop()
                    self.modeling_utterances_ids_train.append(modeling_speaker_utterance_id)
                    modeling_speaker_train_time_acc += \
                        self.utterances_info[modeling_speaker_utterance_id]['wav_len']
                else:
                    self.logger.warning('Not enough data for {}'.format(modeling_speaker_id))
                    break

        # Sort the utterances ids to the natural order
        self.modeling_utterances_ids_train.sort()
        self.modeling_utterances_ids_val.sort()
        self.modeling_utterances_ids_test.sort()

        # Iterate over adaptation speakers ids
        for adaptation_speaker_id in self.adaptation_speakers_ids:
            # Get the list of utterances ids of the speaker
            adaptation_speaker_utterances_ids = [
                utterance_id for utterance_id, utterance in self.utterances_info.items()
                if utterance['speaker_id'] == adaptation_speaker_id
            ]

            # Shuffle the utterances, if required
            if shuffle:
                random.shuffle(adaptation_speaker_utterances_ids)

            # Fill the test split of adaptation
            adaptation_speaker_test_time_acc = 0
            while adaptation_speaker_test_time_acc < self.conf.split['adaptation_test_time_per_speaker']:
                if len(adaptation_speaker_utterances_ids) > 0:
                    adaptation_speaker_utterance_id = adaptation_speaker_utterances_ids.pop()
                    self.adaptation_utterances_ids_test.append(adaptation_speaker_utterance_id)
                    adaptation_speaker_test_time_acc += \
                        self.utterances_info[adaptation_speaker_utterance_id]['wav_len']
                else:
                    self.logger.warning('Not enough data for {}'.format(adaptation_speaker_id))
                    break

            # Fill the validation split of adaptation
            adaptation_speaker_val_time_acc = 0
            while adaptation_speaker_val_time_acc < self.conf.split['adaptation_val_time_per_speaker']:
                if len(adaptation_speaker_utterances_ids) > 0:
                    adaptation_speaker_utterance_id = adaptation_speaker_utterances_ids.pop()
                    self.adaptation_utterances_ids_val.append(adaptation_speaker_utterance_id)
                    adaptation_speaker_val_time_acc += \
                        self.utterances_info[adaptation_speaker_utterance_id]['wav_len']
                else:
                    self.logger.warning('Not enough data for {}'.format(adaptation_speaker_id))
                    break

            # Fill the train split of adaptation
            adaptation_speaker_train_time_acc = 0
            while adaptation_speaker_train_time_acc < self.conf.split['adaptation_train_time_per_speaker']:
                if len(adaptation_speaker_utterances_ids) > 0:
                    adaptation_speaker_utterance_id = adaptation_speaker_utterances_ids.pop()
                    self.adaptation_utterances_ids_train.append(adaptation_speaker_utterance_id)
                    adaptation_speaker_train_time_acc += \
                        self.utterances_info[adaptation_speaker_utterance_id]['wav_len']
                else:
                    self.logger.warning('Not enough data for {}'.format(adaptation_speaker_id))
                    break

        # Sort the utterances ids to the natural order
        self.adaptation_utterances_ids_train.sort()
        self.adaptation_utterances_ids_val.sort()
        self.adaptation_utterances_ids_test.sort()
