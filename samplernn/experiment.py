from .configuration import SampleRNNConfiguration
from .utils import SampleRNNLogger
from .data import SampleRNNData
import os.path
import torch
import soundfile as sf
import re
import shutil


class SampleRNNExperiment:
    """Class used to handle auxiliar procedures related with the execution of an experiment
    """

    name: str
    is_adaptation: bool

    path: str
    logs_path: str
    tensorboard_path: str
    checkpoints_path: str
    generated_samples_path: str

    config_file_path: str
    data_file_path: str
    executions_file_path: str
    checkpoint_base_path: str

    execution = None
    conf: SampleRNNConfiguration
    data: SampleRNNData
    logger: SampleRNNLogger

    checkpoint_format = '{}-modeling-{}.pkl'
    generated_sample_format = '{}-{}-{}-e{}-generated'
    generated_sample_format_pase = '{}-{}-{}-e{}-d{}-generated'

    available_checkpoints = []

    def __init__(self, execution, experiment_name: str):
        """Constructor of the SampleRNNExperiment object

        Args:
            execution (samplernn.SampleRNNExecution): execution object
        """
        self.name = experiment_name
        self.path = os.path.dirname(os.path.dirname(__file__)) + os.sep + 'experiments' + os.sep + self.name + os.sep
        self.logs_path = self.path + 'logs' + os.sep
        self.tensorboard_path = self.path + 'tensorboard' + os.sep
        self.checkpoints_path = self.path + 'checkpoints' + os.sep
        self.generated_samples_path = self.path + 'generated_samples' + os.sep
        self.config_file_path = self.path + 'model.config.yml'
        self.data_file_path = self.path + 'data.split.pkl'
        self.executions_file_path = self.path + 'executions.txt'
        self.checkpoint_base_path = self.checkpoints_path + 'base.pkl'
        self.is_adaptation = os.path.exists(self.checkpoint_base_path)
        self.execution = execution
        self._validate_experiment()

    def _validate_experiment(self):
        """

        """
        # Check that there are correct modeling checkpoints
        if self.available_checkpoints and max(self.available_checkpoints) != len(self.available_checkpoints):
            self.logger.error('Modeling checkpoint files are not valid')
            exit()

    def create(self):
        """Creates the files and folders associated to a new experiment
        """
        os.makedirs(self.logs_path)
        os.makedirs(self.tensorboard_path)
        os.makedirs(self.checkpoints_path)
        os.makedirs(self.generated_samples_path)
        shutil.copyfile(self.execution.model_config, self.config_file_path)
        self.execution.log_execution_header()
        self.logger.info('Experiment "{}" created'.format(self.name))

    def create_adaptation(self):

        # Get the BASE checkpoint path
        base_checkpoint_path = self.execution.experiment_base.checkpoints_path + self.checkpoint_format.format(
            self.execution.experiment_base.name, self.execution.checkpoint_base_epoch)

        # Create the DIRS of the new experiment
        os.makedirs(self.logs_path)
        os.makedirs(self.tensorboard_path)
        os.makedirs(self.checkpoints_path)
        os.makedirs(self.generated_samples_path)

        # Copy config.yml, data.split.pkl and the base checkpoint files
        shutil.copyfile(self.execution.experiment_base.config_file_path, self.config_file_path)
        shutil.copyfile(self.execution.experiment_base.data_file_path, self.data_file_path)
        shutil.copyfile(base_checkpoint_path, self.checkpoint_base_path)

        # Log the process
        self.execution.log_execution_header()
        self.logger.info('Experiment "{}" created from {}'.format(self.name, self.execution.experiment_base_name))

    def load(self):
        """Loads an existing experiment"""
        self.conf = SampleRNNConfiguration(logger=self.logger)
        self.conf.load_file(config_file_path=self.execution.global_config)
        self.conf.load_file(config_file_path=self.config_file_path, initialize_computed_parameters=True)
        self.data = SampleRNNData(conf=self.conf, logger=self.logger)
        self.data.load_data_main(data_main_file_path=self.execution.data_main_path)
        self.data.load_data_split(data_split_file_path=self.data_file_path)
        self._load_available_checkpoints()
        self.logger.info('Experiment "{}" loaded successfully'.format(self.name))

    def _load_available_checkpoints(self):
        if self.is_adaptation:
            self.available_checkpoints.append(0)
        for checkpoint_file in os.listdir(self.checkpoints_path):
            modeling_regex = re.search(r'{}-modeling-(\d+)'.format(self.name), checkpoint_file)
            if modeling_regex:
                self.available_checkpoints.append(int(modeling_regex.group(1)))

    def save_checkpoint(self, epoch_n, train_iteration_n, validation_iteration_n, embedding_state, model_state,
                        optimizer_state, lr_scheduler_state):
        torch.save({
            'epoch_n': epoch_n,
            'train_iteration_n': train_iteration_n,
            'validation_iteration_n': validation_iteration_n,
            'embedding_state': embedding_state,
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'lr_scheduler': lr_scheduler_state
        }, self.checkpoints_path + self.checkpoint_format.format(self.name, epoch_n))

    def load_checkpoint(self, epoch_n):
        checkpoint_path = self.checkpoint_base_path if epoch_n == 0 \
            else self.checkpoints_path + self.checkpoint_format.format(self.name, epoch_n)
        checkpoint_location = ('cuda:{}'.format(self.execution.cuda_device) if self.execution.cuda else 'cpu')
        return torch.load(checkpoint_path, map_location=checkpoint_location)

    def get_generated_sample_name(self, data_info_item, epoch_n, pase_seed_duration=None):
        # Format the name of the utterance
        if pase_seed_duration:
            return self.generated_sample_format_pase.format(
                data_info_item['dataset']['name'],
                data_info_item['speaker']['name'],
                data_info_item['utterance']['name'],
                epoch_n,
                pase_seed_duration
            )
        else:
            return self.generated_sample_format.format(
                data_info_item['dataset']['name'],
                data_info_item['speaker']['name'],
                data_info_item['utterance']['name'],
                epoch_n
            )

    def save_generated_sample(self, data_samples, data_info_item, epoch_n, store_tb, store_file,
                              pase_seed_duration=None):
        # Format the name of the utterance
        generated_sample_name = self.get_generated_sample_name(data_info_item, epoch_n, pase_seed_duration)

        # Store in TensorBoard if required
        if store_tb:
            self.execution.tbx.add_audio(
                tag='Modeling_Epoch-{}_Test/{}'.format(epoch_n, generated_sample_name),
                snd_tensor=data_samples,
                global_step=epoch_n,
                sample_rate=16000
            )

        # Store in the FS if required
        if store_file:
            sf.write(self.generated_samples_path + generated_sample_name + '.wav', data_samples, 16000,
                     subtype='PCM_16')

    def clean_generated_samples(self, data_info, epoch_n, pase_seed_duration=None):

        # Create a new list to store the values
        data_info_new = []

        # Iterate over the data
        for data_info_item in data_info:

            # Format the name of the utterance
            data_info_name = self.get_generated_sample_name(data_info_item, epoch_n, pase_seed_duration)

            # Check if the utterance exists
            if not os.path.exists(self.generated_samples_path + data_info_name + '.wav'):
                data_info_new.append(data_info_item)

        # Return the new list
        return data_info_new

    def clean_generated_conds(self, data_info, epoch_n, pase_seed_duration=None):

        # Create a new list to store the values
        data_info_new = []

        # Iterate over the data
        for data_info_item in data_info:

            # Format the name of the utterance
            data_info_name = self.get_generated_sample_name(data_info_item, epoch_n, pase_seed_duration)

            # Verify that the file exists
            assert os.path.exists(self.generated_samples_path + data_info_name + '.wav')

            # Add the item to the list if conds does not exist
            if not os.path.exists(self.generated_samples_path + data_info_name + '.cc') or \
                    not os.path.exists(self.generated_samples_path + data_info_name + '.fv') or \
                    not os.path.exists(self.generated_samples_path + data_info_name + '.lf0'):
                data_info_new.append(data_info_item)

        # Return the cleaned list
        return data_info_new

    def data_info_to_paths(self, data_info, epoch_n, pase_seed_duration=None):

        # Create a new list to store the path
        data_info_paths = []

        # Iterate over the data
        for data_info_item in data_info:

            # Format the name of the utterance
            data_info_name = self.get_generated_sample_name(data_info_item, epoch_n, pase_seed_duration)

            # Append the item to the list
            data_info_paths.append(self.generated_samples_path + data_info_name + '.wav')

        # Return the list of paths
        return data_info_paths

