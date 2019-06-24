from .configuration import SampleRNNConfiguration
from .dataset import SampleRNNDataset
from .utils import SampleRNNQuantizer
from .data import SampleRNNData
from .execution import SampleRNNExecution
import torch
import numpy as np
import random
from torch.utils.data import DataLoader


class SampleRNNDataLoader(DataLoader):
    """

    """
    execution: SampleRNNExecution
    conf: SampleRNNConfiguration
    dataset: SampleRNNDataset
    quantizer: SampleRNNQuantizer

    is_adaptation: bool
    split: str

    dataset_iterator = None
    buffer = []
    reset_buffer = []
    no_more_samples_in_batch: bool

    require_full_batch: bool

    batch_size: int
    utterance_conds_size: int
    speaker_conds_size: int
    normalize_conds: bool

    _exclude_utterances_ids = []

    def __init__(self, execution: SampleRNNExecution, quantizer: SampleRNNQuantizer, is_adaptation: bool, split: str):
        self.execution = execution
        self.conf = execution.experiment.conf
        self.quantizer = quantizer
        self.is_adaptation = is_adaptation
        self.split = split
        self.batch_size = self.conf.training['batch_size']
        self.utterance_conds_size = self.conf.conditionants['utterance_size']
        self.speaker_conds_size = self.conf.conditionants['speaker_size']
        self.normalize_conds = True
        self.dataset = SampleRNNDataset(
            execution=self.execution,
            quantizer=self.quantizer,
            normalize_conds=self.normalize_conds,
            is_adaptation=self.is_adaptation,
            split=self.split
        )
        self.require_full_batch = self.execution.command == 'train'

        # Call parent constructor
        super().__init__(
            dataset=self.dataset,
            batch_size=self.batch_size
        )

    def __iter__(self):

        # Reset parameters
        self._reset_parameters()

        while True:
            # Prepare the buffers for the next iteration
            self._prepare_buffers()

            # Fill Data
            self._fill_data()

            # Check parallelism
            if self.require_full_batch and not all(self.buffer) or not any(self.buffer):
                break

            # Handle Train and Validation splits
            yield self._get_tensors()

    def get_random_utterances(self, n_samples: int):

        # Create a dict {speaker_id: [utterance_ids]}
        speakers_utterances_ids = {}
        for utterance_id in self.dataset.utterances_ids:
            utterance_speaker_id = self.dataset.data.utterances_info[utterance_id]['speaker_id']
            if utterance_speaker_id not in speakers_utterances_ids:
                speakers_utterances_ids[utterance_speaker_id] = [self.dataset.utterances_ids.index(utterance_id)]
            else:
                speakers_utterances_ids[utterance_speaker_id].append(self.dataset.utterances_ids.index(utterance_id))

        # Iterate over the
        utterances = []
        for speaker_id, utterance_ids in speakers_utterances_ids.items():
            selected_utterance_ids = random.sample(utterance_ids, n_samples)
            for selected_random_sample in selected_utterance_ids:
                utterances.append(self.dataset.__getitem__(selected_random_sample))

        # Get maximum sequence length
        utterance_max_seq_len = max([utterance[1].shape[0] for utterance in utterances])

        # Initialize Tensors
        samples = torch.zeros(len(utterances), (utterance_max_seq_len + 1) * self.conf.architecture['frame_size'])
        conds_speakers = torch.zeros(len(utterances), self.speaker_conds_size)
        conds_utterances = torch.zeros(len(utterances), utterance_max_seq_len, self.utterance_conds_size)
        model_reset = torch.ones(len(utterances), dtype=torch.uint8)
        info = [None] * len(utterances)

        # Move it to GPU, if desired
        if self.execution.cuda:
            samples = samples.cuda()
            conds_utterances = conds_utterances.cuda()
            conds_speakers = conds_speakers.cuda()

        # Fill the Tensors with Data
        for utterance_index, utterance in enumerate(utterances):
            samples[utterance_index, :utterance[0].shape[0]] = torch.Tensor(utterance[0])
            conds_speakers[utterance_index, :] = torch.Tensor(utterance[2])
            conds_utterances[utterance_index, :utterance[1].shape[0], :] = torch.Tensor(utterance[1])
            info[utterance_index] = utterance[3]

        return samples, None, conds_speakers, conds_utterances, model_reset, info

    def get_random_chunks(self, speakers, chunk_length, fixed_start=False):
        random_chunks = np.zeros((len(speakers), 16000 * chunk_length))
        speaker_ids = []
        for i, speaker in enumerate(speakers):
            if speaker != 0:
                speaker_id = self.dataset.data.datasets_info[speaker['dataset_id']]['speakers_prefix'] + speaker['name']
                random_chunks[i, :] = self.dataset.get_random_chunk(speaker_id, 16000 * chunk_length, fixed_start)
        return torch.Tensor(random_chunks)

    def get_data_info(self, speaker_id=None, max_samples=np.inf):
        if speaker_id:
            return [data_item[3] for data_item in self.dataset if speaker_id == data_item[3]['utterance']['speaker_id']]
        else:
            return [self.dataset[item_index][3] for item_index in range(min(self.dataset.__len__(), max_samples))]

    def set_exclude_utterances(self, include_data_info):
        all_uterances = self.get_data_info()
        exclude_utterances = [utterance for utterance in all_uterances if utterance not in include_data_info]
        self._exclude_utterances_ids = [SampleRNNData.get_utterance_id(eut) for eut in exclude_utterances]

    def _reset_parameters(self):
        self.dataset.shuffle_utterances()
        self.dataset_iterator = iter(self.dataset)
        self.buffer = [None] * self.batch_size
        self.reset_buffer = [None] * self.batch_size
        self.no_more_samples_in_batch = False

    def _prepare_buffers(self):
        # Lala
        for buffer_index, buffer_item in enumerate(self.buffer):

            # If the buffer item is already None, continue the loop
            if buffer_item is None:
                continue

            # Set the reset flag to false
            self.reset_buffer[buffer_index] = False

            # If there are not enoguh samples, set it to None
            if buffer_item[0].shape[0] == self.conf.architecture['frame_size'] or buffer_item[1].shape[0] == 0:
                self.buffer[buffer_index] = None
                self.reset_buffer[buffer_index] = None

    def _fill_data(self):

        # Iterate until fill the buffer or not more samples in the batch
        while not all(self.buffer) and not self.no_more_samples_in_batch:
            # Handle no more samples in batch
            try:
                # Get indexes of the Nones of the buffer
                none_indexes = [i for i, x in enumerate(self.buffer) if x is None]
                none_index = random.choice(none_indexes)

                # Store the utterance in the Dataset
                next_item = list(next(self.dataset_iterator))

                # Check that the element is not excluded
                utterance_id = SampleRNNData.get_utterance_id(next_item[3])
                if utterance_id not in self._exclude_utterances_ids:
                    self.buffer[none_index] = next_item
                    self.reset_buffer[none_index] = True

            # If no more samples in batch, set the flag
            except StopIteration:
                self.no_more_samples_in_batch = True

    def _get_tensors(self):

        # Compute placeholders size if the split is train or validation
        if self.split in ['train', 'validation']:
            samples_len = self.conf.architecture['receptive_field'] + self.conf.architecture['frame_size'] - 1
            samples_target_len = self.conf.architecture['receptive_field']
            utterance_conds_len = self.conf.architecture['sequence_length']


        # Compute placeholders size if the split is test
        elif self.split in ['test']:
            samples_len = max([buffer_item[0].shape[0] for buffer_item in self.buffer if buffer_item is not None])
            samples_target_len = samples_len - self.conf.architecture['frame_size']
            utterance_conds_len = max(
                [buffer_item[1].shape[0] for buffer_item in self.buffer if buffer_item is not None]
            )

        # Initialize Tensors
        samples = torch.zeros(self.batch_size, samples_len)
        samples_target = torch.zeros(self.batch_size, samples_target_len)
        conds_speakers = torch.zeros(self.batch_size, self.speaker_conds_size)
        conds_utterances = torch.zeros(self.batch_size, utterance_conds_len, self.utterance_conds_size)
        info = [None] * self.batch_size
        model_reset = torch.zeros(self.batch_size, dtype=torch.uint8)

        # Move it to GPU, if desired
        if self.execution.cuda:
            samples = samples.cuda()
            samples_target = samples_target.cuda()
            conds_speakers = conds_speakers.cuda()
            conds_utterances = conds_utterances.cuda()
            model_reset = model_reset.cuda()

        # Prepare model_reset
        for reset_buffer_index, reset_buffet_item in enumerate(self.reset_buffer):

            # Assign a 2 if is None (not a valid audio in that position
            if reset_buffet_item is None:
                model_reset[reset_buffer_index] = 2

            # Assign a 1 if True
            elif reset_buffet_item is True:
                model_reset[reset_buffer_index] = 1

            # Assign a 0 if False
            elif reset_buffet_item is False:
                model_reset[reset_buffer_index] = 0

        # Iterate over buffer
        for buffer_index, buffer_item in enumerate(self.buffer):

            # Check if buffer item is not None
            if buffer_item is None:
                continue

            # Fill shared parameters between train, validation and test
            conds_speakers[buffer_index, :] = torch.Tensor(buffer_item[2])
            info[buffer_index] = buffer_item[3]

            # Handle Train and Validation splits
            if self.split in ['train', 'validation']:

                # Fill tensors
                samples[buffer_index, :] = torch.Tensor(buffer_item[0][:(
                        self.conf.architecture['receptive_field'] + self.conf.architecture['frame_size'] - 1
                )])
                conds_utterances[buffer_index, :, :] = torch.Tensor(
                    buffer_item[1][:self.conf.architecture['sequence_length'], :]
                )
                samples_target[buffer_index, :] = torch.Tensor(
                    buffer_item[0][self.conf.architecture['frame_size']:(self.conf.architecture['receptive_field'] +
                                                                         self.conf.architecture['frame_size'])]
                )

                # Clean samples from the utterance
                buffer_item[0] = buffer_item[0][self.conf.architecture['receptive_field']:]
                buffer_item[1] = buffer_item[1][self.conf.architecture['sequence_length']:, :]

            # Handle Test split
            elif self.split == 'test':

                # Fill tensors
                samples[buffer_index, :buffer_item[0].shape[0]] = torch.Tensor(buffer_item[0])
                conds_utterances[buffer_index, :buffer_item[1].shape[0], :] = torch.Tensor(buffer_item[1])
                samples_target[buffer_index, :buffer_item[0].shape[0] - self.conf.architecture[
                    'frame_size']] = torch.Tensor(buffer_item[0][self.conf.architecture['frame_size']:])

                # Clean samples from the utterance
                buffer_item[0] = buffer_item[0][-self.conf.architecture['frame_size']:]
                buffer_item[1] = buffer_item[1][buffer_item[1].shape[0]:, :]

        # Return the values
        return samples, samples_target, conds_speakers, conds_utterances, model_reset, info
