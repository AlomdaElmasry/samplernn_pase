import tensorboardX
import os
import torch
from .utils import SampleRNNLogger
from .experiment import SampleRNNExperiment


class SampleRNNExecution:
    """Class to handle a SampleRNN Execution"""

    command: str
    experiment_name: str
    checkpoint_epoch: int

    experiment_base_name: str
    checkpoint_base_epoch: int

    global_config: str
    split_config: str
    model_config: str
    data_main_path: str

    seed: int
    cuda: bool
    cuda_device: int
    parallel: bool
    verbose: bool

    _name: str
    _log_file_path: str = None
    _tensorboard_folder_path: str

    experiment = None
    experiment_base = None
    logger: SampleRNNLogger
    tbx: tensorboardX.SummaryWriter

    def __init__(self, command: str, experiment_name: str, checkpoint_epoch: int, experiment_base_name: str,
                 checkpoint_base_epoch: int, global_config: str, split_config: str,
                 model_config: str, data_main_path: str, seed: int, cuda: bool, cuda_device: int, parallel: bool,
                 verbose: str):

        # Store Parameters
        self.command = command
        self.experiment_name = experiment_name
        self.checkpoint_epoch = checkpoint_epoch
        self.experiment_base_name = experiment_base_name
        self.checkpoint_base_epoch = checkpoint_base_epoch
        self.global_config = global_config
        self.split_config = split_config
        self.model_config = model_config
        self.data_main_path = data_main_path
        self.seed = seed
        self.cuda = cuda
        self.cuda_device = cuda_device
        self.parallel = parallel
        self.verbose = verbose

        # Initialize Experiment
        if self.experiment_name:
            self.experiment = SampleRNNExperiment(execution=self, experiment_name=self.experiment_name)

        # Initialize Base Experiment
        if self.experiment_base_name:
            self.experiment_base = SampleRNNExperiment(execution=self, experiment_name=self.experiment_base_name)

        # Initialize self._name
        self._init_name()

        # Initialize Logger
        self._init_logger()

        # Initialize TensorBoard
        self._init_tensorboard()

        # Assign Logger to the experiment
        if self.experiment_name:
            self.experiment.logger = self.logger
        if self.experiment_base_name:
            self.experiment_base.logger = self.logger

        # Validate Execution
        if self.experiment_name:
            self._validate_execution()

    def _validate_execution(self):
        # Validate CUDA
        if self.cuda and not torch.cuda.is_available():
            self.cuda = False
            self.logger.warning('CUDA required but not available')

        # Validate CUDA DEVICE
        if self.cuda_device + 1 > torch.cuda.device_count():
            self.cuda_device = 0
            self.logger.warning('Required GPU that does not exist')

        # Validate Multi-GPU
        if self.parallel and torch.cuda.device_count() < 2:
            self.parallel = False
            self.logger.warning('Multi-GPU required but not available')

    def _init_name(self):
        self._name = self.command + ('_epoch_' + str(self.checkpoint_epoch) if self.checkpoint_epoch else '')

    def _init_logger(self):

        # Initialize self._log_file_path only with the following commands
        if self.command not in ['init', 'init_adaptation']:

            # Define variable
            execution_suffix = ''

            # If the log already exists, append a number at the end
            i = 1
            while os.path.exists(self.experiment.logs_path + self._name + execution_suffix + '.log'):
                execution_suffix = '_' + str(i)
                i += 1

            # Set the parameter
            self._log_file_path = self.experiment.logs_path + self._name + execution_suffix + '.log'

            # Create the log file
            open(self._log_file_path, 'a').close()

        # Initialize the Logger object
        self.logger = SampleRNNLogger(log_file_path=self._log_file_path, verbose=self.verbose)

    def _init_tensorboard(self):

        # Initialize self._tensorboard_folder_path only with the following commands
        if self.command not in ['init', 'init_adaptation']:
            self.tbx = tensorboardX.SummaryWriter(self.experiment.tensorboard_path)

    def log_execution_header(self):
        with open(self.experiment.executions_file_path, 'a') as executions_file:
            executions_file.write(
                '=' * 19 + '|' + '=' * 19 + '|' + '=' * 19 + '|' + '=' * 19 + '|' + '=' * 19 + '|' + "\n"
            )
            executions_file.write('Command' + 12 * ' ' + '|')
            executions_file.write(' ' + 'Experiment Name' + 3 * ' ' + '|')
            executions_file.write(' ' + 'Epoch Number' + 6 * ' ' + '|')
            executions_file.write(' ' + 'Verbose' + 11 * ' ' + '|' + "\n")
            executions_file.write(
                '=' * 19 + '|' + '=' * 19 + '|' + '=' * 19 + '|' + '=' * 19 + '|' + '=' * 19 + '|' + "\n"
            )

    def log_execution_entry(self):
        with open(self.experiment.executions_file_path, 'a') as executions_file:
            executions_file.write(self.command + (20 - len(self.command) - 1) * ' ' + '|')
            executions_file.write(' ' + self.experiment.name + (20 - len(self.experiment.name) - 2) * ' ' + '|')
            executions_file.write(
                ' ' + str(self.checkpoint_epoch) + (20 - len(str(self.checkpoint_epoch)) - 2) * ' ' + '|')
            executions_file.write(' ' + str(self.verbose) + (20 - len(str(self.verbose)) - 2) * ' ' + '|' + "\n")
