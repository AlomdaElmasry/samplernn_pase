from .configuration import SampleRNNConfiguration
from .execution import SampleRNNExecution
from .experiment import SampleRNNExperiment
from .loader import SampleRNNDataLoader
from .model import SampleRNNModel
from .utils import SampleRNNLogger, SampleRNNQuantizer, SampleRNNAhocoder
import ahoproc_tools.error_metrics
import concurrent.futures
import numpy as np
import math
import tensorboardX
import torch
import random
import pase.frontend


class SampleRNN:
    """Handler for the SampleRNN"""

    execution: SampleRNNExecution
    experiment: SampleRNNExperiment
    conf: SampleRNNConfiguration
    quantizer: SampleRNNQuantizer
    logger: SampleRNNLogger

    embedding_layer: torch.nn.Embedding
    pase_encoder: pase.frontend.WaveFe
    model: SampleRNNModel
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.StepLR

    train_data_loader: SampleRNNDataLoader
    val_data_loader: SampleRNNDataLoader
    test_data_loader: SampleRNNDataLoader

    is_adaptation: bool
    epoch_n: int = 0
    train_iteration_n: int = 0
    validation_iteration_n: int = 0

    def __init__(self, execution: SampleRNNExecution):

        self.execution = execution
        self.experiment = execution.experiment
        self.conf = execution.experiment.conf
        self.quantizer = SampleRNNQuantizer(conf=self.experiment.conf)
        self.logger = execution.logger

        # Initialize Seed if set
        if self.execution.seed is not None:
            np.random.seed(self.execution.seed)
            random.seed(self.execution.seed)
            torch.manual_seed(self.execution.seed)
            if self.execution.cuda:
                torch.cuda.manual_seed_all(self.execution.seed)

        # Create the Embedding layer
        if self.conf.conditionants['speaker_type'] == 'embedding':
            self.embedding_layer = torch.nn.Embedding(
                num_embeddings=len(self.experiment.data.speakers_info),
                embedding_dim=self.conf.conditionants['speaker_embedding_size']
            )
            self.pase_encoder = None

        # Handle the use of PASE as encoder
        elif self.conf.conditionants['speaker_type'] in ['pase_seed', 'pase_trained']:
            self.embedding_layer = None
            self.pase_encoder = pase.frontend.wf_builder(self.conf.pase['config_file_path'])
            self.pase_encoder.load_pretrained(self.conf.pase['trained_model_path'], load_last=True, verbose=True)

        # Create new model with the desired configuration and Quantizer
        self.model = SampleRNNModel(
            conf=self.experiment.conf,
            quantizer=self.quantizer,
            conds_linguistic_n=[
                len(self.experiment.data.utterances_conds_linguistic_categories['phonemes']),
                len(self.experiment.data.utterances_conds_linguistic_categories['vowels']),
                len(self.experiment.data.utterances_conds_linguistic_categories['gpos']),
                len(self.experiment.data.utterances_conds_linguistic_categories['tobi'])
            ]
        )

        # Create new Optimizer with the desired configuration
        params_to_optimize = list(self.model.parameters())
        if self.conf.conditionants['speaker_type'] == 'embedding':
            params_to_optimize += list(self.embedding_layer.parameters())
        if self.conf.conditionants['speaker_type'] == 'pase_trained':
            params_to_optimize += list(self.pase_encoder.parameters())
        self.optimizer = torch.optim.Adam(params=params_to_optimize, lr=self.conf.training['lr'])

        # Create the Scheduler
        if self.conf.training['lr_scheduler']:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min',
                patience=self.conf.training['lr_scheduler_patience'],
                factor=self.conf.training['lr_scheduler_factor']
            )

        # Check if we want to use GPU
        if self.execution.cuda:

            # Set the CUDA devic
            torch.cuda.set_device(self.execution.cuda_device)

            # Move Embedding and Model to GPU
            self.model = self.model.cuda()
            if self.conf.conditionants['speaker_type'] == 'embedding':
                self.embedding_layer = self.embedding_layer.cuda()
            elif self.conf.conditionants['speaker_type'] in ['pase_seed', 'pase_trained']:
                self.pase_encoder = self.pase_encoder.cuda()

            # Move each state to CUDA
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        # Check Parallelism
        if self.execution.parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.conf.training['batch_size'] = self.conf.training['batch_size'] * torch.cuda.device_count()

        # Create the Loaders
        self.train_data_loader = SampleRNNDataLoader(
            execution=self.execution,
            quantizer=self.quantizer,
            is_adaptation=self.experiment.is_adaptation,
            split='train'
        )
        self.val_data_loader = SampleRNNDataLoader(
            execution=self.execution,
            quantizer=self.quantizer,
            is_adaptation=self.experiment.is_adaptation,
            split='validation'
        )
        self.test_data_loader = SampleRNNDataLoader(
            execution=self.execution,
            quantizer=self.quantizer,
            is_adaptation=self.experiment.is_adaptation,
            split='test'
        )

    def train(self):

        # Load checkpoint if required
        if len(self.experiment.available_checkpoints) > 0:
            self._load_checkpoint(epoch_n=max(self.experiment.available_checkpoints))

        # Iterate over the epochs
        for self.epoch_n in range(self.epoch_n + 1, self.conf.training['max_epochs'] + 1):

            # Initialize lists
            train_losses = []
            val_losses = []

            # Log entrance in batch
            self.experiment.logger.info('Initiating epoch number {}'.format(self.epoch_n))

            # Set the model in train mode
            self.model.train()

            # Training Data
            for self.train_iteration_n, data in enumerate(self.train_data_loader, self.train_iteration_n + 1):
                train_loss = self._step_train(data=data)
                train_losses.append(train_loss)

            # Log iteration loss
            self.experiment.logger.info('Finished training of epoch {}. Starting validation...'.format(self.epoch_n))

            # Validation Data
            for self.validation_iteration_n, data in enumerate(self.val_data_loader, self.validation_iteration_n + 1):
                val_loss = self._step_validation(data=data)
                val_losses.append(val_loss)

            # Log iteration loss
            self.experiment.logger.info(
                'Finished validation of epoch {}. Computing mean losses and adjusting LR...'.format(self.epoch_n)
            )

            # Compute mean epoch losses
            train_mean_loss = np.mean(train_losses)
            val_mean_loss = np.mean(val_losses)

            # Log LR
            self.execution.tbx.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.epoch_n)

            # Step the LR Scheduler
            if self.conf.training['lr_scheduler']:
                self.lr_scheduler.step(val_mean_loss)
                self.experiment.logger.info('LR adjusted. Generating audio samples from train...'.format(self.epoch_n))

            # Store Mean Losses in TB
            self.execution.tbx.add_scalar('loss/train', train_mean_loss, self.epoch_n)
            self.execution.tbx.add_scalar('loss/validation', val_mean_loss, self.epoch_n)

            # Save epoch checkpoint
            self.experiment.save_checkpoint(
                epoch_n=self.epoch_n,
                train_iteration_n=self.train_iteration_n,
                validation_iteration_n=self.validation_iteration_n,
                embedding_state=(self.embedding_layer.state_dict() if self.embedding_layer else None),
                pase_state=(self.pase_encoder.state_dict() if self.pase_encoder else None),
                model_state=(
                    self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel)
                    else self.model.state_dict()
                ),
                optimizer_state=self.optimizer.state_dict(),
                lr_scheduler_state=self.lr_scheduler.state_dict()
            )

            # Log iteration loss
            self.experiment.logger.info('Checkpoint of epoch {} saved. Generating Samples...'.format(self.epoch_n))

    def test(self):

        # Set the Loader in Generation Mode and Model in eval()
        self.model.eval()

        # Iterate over the list of epochs to test
        for test_epoch in self.execution.checkpoint_epoch:
            # Load checkpoint
            self._load_checkpoint(epoch_n=test_epoch)

            # Get a list of the Test Utterances and generated test utterances
            cc_mcd, f0_rmse, vu_afpr = self._step_test(
                data_info=self.test_data_loader.get_data_info()
            )

            # Log results in TensorBoard
            self.execution.tbx.add_scalar('test/mcd', cc_mcd, self.epoch_n)
            self.execution.tbx.add_scalar('test/f0_rmse', f0_rmse, self.epoch_n)
            self.execution.tbx.add_scalar('test/accuracy', vu_afpr[0], self.epoch_n)
            self.execution.tbx.add_scalar('test/fmeasure', vu_afpr[1], self.epoch_n)
            self.execution.tbx.add_scalar('test/precision', vu_afpr[2], self.epoch_n)
            self.execution.tbx.add_scalar('test/recall', vu_afpr[3], self.epoch_n)

    def test_speaker_by_seed(self, speaker_id, seed_durations):

        # Set the Loader in Generation Mode and Model in eval()
        self.model.eval()

        # Check if speaker_id = ["*"] and substitute it with a list of all speakers
        if speaker_id == ["all"]:
            if self.experiment.is_adaptation:
                speaker_id = self.experiment.data.adaptation_speakers_ids
            else:
                speaker_id = self.experiment.data.modeling_speakers_ids

        # Iterate over the list of epochs to test
        for test_epoch in self.execution.checkpoint_epoch:

            # Load checkpoint
            self._load_checkpoint(epoch_n=test_epoch)

            # Iterate over the SEED durations
            for seed_duration in seed_durations:

                # Set the SEED duration in the test dataset
                self.test_data_loader.dataset.set_pase_seed_duration(seed_duration)

                # If perform mean between speakers
                mcd_mean = []
                f0_rmse_mean = []

                # Iterate over the list of speakers to test
                for test_speaker_id in speaker_id:

                    # Check that the speaker exists
                    if self.experiment.is_adaptation:
                        assert test_speaker_id in self.experiment.data.adaptation_speakers_ids
                    else:
                        assert test_speaker_id in self.experiment.data.modeling_speakers_ids

                    # Peform the step
                    cc_mcd, f0_rmse, vu_afpr = self._step_test(
                        data_info=self.test_data_loader.get_data_info(speaker_id=test_speaker_id),
                        speaker_id=test_speaker_id,
                        pase_seed_duration=seed_duration
                    )

                    # Store the results in the aux variable
                    mcd_mean.append(cc_mcd)
                    f0_rmse_mean.append(f0_rmse)

                    # Log the speaker values
                    self.execution.tbx.add_scalar(
                        'seed-dur/e{}/{}/mcd'.format(test_epoch, test_speaker_id), cc_mcd, seed_duration
                    )
                    self.execution.tbx.add_scalar(
                        'seed-dur/e{}/{}/f0_rmse'.format(test_epoch, test_speaker_id), f0_rmse, seed_duration
                    )

                # Store the mean values
                self.execution.tbx.add_scalar(
                    'seed-dur/e{}/mean/mcd'.format(test_epoch), np.mean(mcd_mean), seed_duration
                )
                self.execution.tbx.add_scalar(
                    'seed-dur/e{}/mean/f0_rmse'.format(test_epoch), np.mean(f0_rmse_mean), seed_duration
                )

    def infer(self, n_samples):
        # Iterate over the list of epochs to test
        for infer_epoch in self.execution.checkpoint_epoch:
            # Load checkpoint
            self._load_checkpoint(epoch_n=infer_epoch)

            # Set the model in test mode
            self.model.eval()

            # Call aux function
            self._step_infer(
                data=self.test_data_loader.get_random_utterances(n_samples),
                store_tb=True,
                store_file=True
            )

    def _load_checkpoint(self, epoch_n):
        checkpoint_state = self.experiment.load_checkpoint(epoch_n=epoch_n)
        self.model.load_state_dict(checkpoint_state['model_state'])
        if checkpoint_state['embedding_state']:
            self.embedding_layer.load_state_dict(checkpoint_state['embedding_state'])
        if checkpoint_state['pase_state']:
            self.pase_encoder.load_state_dict(checkpoint_state['pase_state'])
        if epoch_n != 0:
            self.optimizer.load_state_dict(checkpoint_state['optimizer_state'])
            self.lr_scheduler.load_state_dict(checkpoint_state['lr_scheduler'])
            self.epoch_n = checkpoint_state['epoch_n']
            self.train_iteration_n = checkpoint_state['train_iteration_n']
            self.validation_iteration_n = checkpoint_state['train_iteration_n']
        self.logger.info('Checkpoint {} loaded succesfully'.format(epoch_n))

    def _step_train(self, data):
        # Set gradients to zero in the optimizer
        self.optimizer.zero_grad()

        # Decompose data
        data_samples, data_samples_target, data_conds_speakers, data_conds_utterances, data_model_reset, data_info = \
            data

        # Quantize the samples
        data_samples = self.quantizer.quantize(data_samples)
        data_samples_target = self.quantizer.quantize(data_samples_target)

        # Use one-hot embedding to identify each speaker
        if self.conf.conditionants['speaker_type'] == 'embedding':
            data_speakers_ids = torch.LongTensor([data_info_item['speaker']['index'] if data_info_item is not None
                                                  else 0 for data_info_item in data_info])
            if self.execution.cuda:
                data_speakers_ids = data_speakers_ids.cuda()
            data_conds_speakers = self.embedding_layer(data_speakers_ids)

        # Use PASE to identify each speaker
        elif self.conf.conditionants['speaker_type'] == 'pase_seed':
            speakers = [data_info_item['speaker'] if data_info_item is not None
                        else 0 for data_info_item in data_info]
            pase_chunks = self.val_data_loader.get_random_chunks(speakers, 16000).unsqueeze(1)
            if self.execution.cuda:
                pase_chunks = pase_chunks.cuda()
            with torch.no_grad():
                pase_output = self.pase_encoder(pase_chunks)
            data_conds_speakers = torch.mean(pase_output, dim=2)
            data_conds_speakers.detach()

        # Use PASE to identify each speaker and fine-tune it end-to-end
        elif self.conf.conditionants['speaker_type'] == 'pase_trained':
            speakers = [data_info_item['speaker'] if data_info_item is not None
                        else 0 for data_info_item in data_info]
            pase_chunks = self.val_data_loader.get_random_chunks(speakers, 16000).unsqueeze(1)
            if self.execution.cuda:
                pase_chunks = pase_chunks.cuda()
            pase_output = self.pase_encoder(pase_chunks)
            data_conds_speakers = torch.mean(pase_output, dim=2)

        # Propagate through the model
        data_samples_predicted = self.model(data_samples, data_conds_speakers, data_conds_utterances, data_model_reset)

        # Clean Invalid indexes of the batch
        data_samples_target = data_samples_target[data_model_reset != 2]
        data_samples_predicted = data_samples_predicted[data_model_reset != 2]

        # Compute the loss
        loss = torch.nn.functional.nll_loss(
            data_samples_predicted.view(-1, self.quantizer.q_levels),
            data_samples_target.view(-1)
        )

        # Backprop
        loss.backward()

        # Optimizer learnable parameters
        self.optimizer.step()

        # Log iteration loss
        self.logger.info(
            'Training: Epoch {}, Iteration {}, Loss {}'.format(self.epoch_n, self.train_iteration_n, loss)
        )

        # Return loss
        return loss.item()

    def _step_validation(self, data):
        # Decompose data
        data_samples, data_samples_target, data_conds_speakers, data_conds_utterances, data_model_reset, data_info = \
            data

        # Quantize the samples
        data_samples = self.quantizer.quantize(data_samples)
        data_samples_target = self.quantizer.quantize(data_samples_target)

        # Use one-hot embedding to identify each speaker
        if self.conf.conditionants['speaker_type'] == 'embedding':
            data_speakers_ids = torch.LongTensor([data_info_item['speaker']['index'] if data_info_item is not None
                                                  else 0 for data_info_item in data_info])
            if self.execution.cuda:
                data_speakers_ids = data_speakers_ids.cuda()
            data_conds_speakers = self.embedding_layer(data_speakers_ids)

        # Use PASE to identify each speaker
        elif self.conf.conditionants['speaker_type'] in ['pase_seed', 'pase_trained']:
            speakers = [data_info_item['speaker'] if data_info_item is not None
                        else 0 for data_info_item in data_info]
            pase_chunks = self.val_data_loader.get_random_chunks(speakers, 16000).unsqueeze(1)
            if self.execution.cuda:
                pase_chunks = pase_chunks.cuda()
            with torch.no_grad():
                pase_output = self.pase_encoder(pase_chunks)
            data_conds_speakers = torch.mean(pase_output, dim=2)

        # Propagate validation samples through the model
        with torch.no_grad():
            data_samples_predicted = self.model(data_samples, data_conds_speakers, data_conds_utterances,
                                                data_model_reset)

        # Clean Invalid indexes of the batch
        data_samples_target = data_samples_target[data_model_reset != 2]
        data_samples_predicted = data_samples_predicted[data_model_reset != 2]

        # Compute the validation loss
        loss = torch.nn.functional.cross_entropy(
            data_samples_predicted.view(-1, self.quantizer.q_levels),
            data_samples_target.view(-1)
        )

        # Log iteration loss
        self.experiment.logger.info(
            'Validation: Epoch {}, Iteration {}, Loss {}'.format(self.epoch_n, self.validation_iteration_n, loss)
        )

        # Return loss
        return loss.item()

    def _step_test(self, data_info, speaker_id=None, pase_seed_duration=None):
        """

        Args:
            data_info:
            pase_seed_duration:

        Returns:

        """

        # Remove items that have already been generated
        data_info_cleaned_samples = self.experiment.clean_generated_samples(
            data_info=data_info,
            epoch_n=self.epoch_n,
            pase_seed_duration=pase_seed_duration
        )

        # Exclude them from the dataset
        self.test_data_loader.set_exclude_utterances(include_data_info=data_info_cleaned_samples)

        # First, we generate those samples which do not exist yet
        for data in self.test_data_loader:
            self._step_infer(data, store_tb=False, store_file=True, pase_seed_duration=pase_seed_duration)

        # Clean data_info from already created conds and get a list of paths of the utterances to generate
        data_info_cleaned_conds = self.experiment.clean_generated_conds(
            data_info=data_info,
            epoch_n=self.epoch_n,
            pase_seed_duration=pase_seed_duration
        )
        data_info_paths = self.experiment.data_info_to_paths(
            data_info=data_info,
            epoch_n=self.epoch_n,
            pase_seed_duration=pase_seed_duration
        )

        # Generate Acoustic Conds and get a list of the
        SampleRNNAhocoder.generate_acoustic_conds(utterances_paths=data_info_paths)

        # Compute and return measures
        return SampleRNNAhocoder.compute_objective_metrics(data_info, self.experiment, self.epoch_n, pase_seed_duration)

    def _step_infer(self, data, store_tb, store_file, pase_seed_duration=None):
        # Decompose data
        data_samples, _, data_conds_speakers, data_conds_utterances, data_model_reset, data_info = data

        # Use one-hot embedding to identify each speaker
        if self.conf.conditionants['speaker_type'] == 'embedding':
            data_speakers_ids = torch.LongTensor([data_info_item['speaker']['index'] if data_info_item is not None
                                                  else 0 for data_info_item in data_info])
            if self.execution.cuda:
                data_speakers_ids = data_speakers_ids.cuda()
            data_conds_speakers = self.embedding_layer(data_speakers_ids)

        # Use PASE to identify each speaker
        elif self.conf.conditionants['speaker_type'] in ['pase_seed', 'pase_trained']:
            speakers = [data_info_item['speaker'] if data_info_item is not None
                        else 0 for data_info_item in data_info]
            pase_chunks = self.val_data_loader.get_random_chunks(speakers, 16000, fixed_start=True).unsqueeze(1)
            if self.execution.cuda:
                pase_chunks = pase_chunks.cuda()
            with torch.no_grad():
                pase_output = self.pase_encoder(pase_chunks)
            data_conds_speakers = torch.mean(pase_output, dim=2)

        # Propagate through the model
        with torch.no_grad():
            data_samples_predicted = self.model(None, data_conds_speakers, data_conds_utterances, data_model_reset)

        # Remove BAD Samples
        data_samples_predicted = data_samples_predicted[data_model_reset != 2]
        data_info = [data_info_item for data_info_item_index, data_info_item in enumerate(data_info) if
                     data_model_reset[data_info_item_index] != 2]

        # Dequentize the samples
        data_samples_predicted = self.quantizer.dequantize(data_samples_predicted)

        # Iterate over the generated samples
        for data_samples_predicted_item, data_info_item in zip(data_samples_predicted, data_info):
            data_samples_predicted_item = data_samples_predicted_item[:data_info_item['utterance']['wav_len']]
            self.experiment.save_generated_sample(
                data_samples=data_samples_predicted_item.cpu(),
                data_info_item=data_info_item,
                epoch_n=self.epoch_n,
                store_tb=store_tb,
                store_file=store_file,
                pase_seed_duration=pase_seed_duration
            )
            self.experiment.logger.info('Utterance {} generated'.format(data_info_item['utterance']['name']))
