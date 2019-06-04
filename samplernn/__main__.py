from . import SampleRNN
from .experiment import SampleRNNExperiment
from .execution import SampleRNNExecution
from .utils import SampleRNNLogger
from .data import SampleRNNData
from .configuration import SampleRNNConfiguration
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Run SampleRNN')

# Add arguments to the main parser
parser.add_argument('--global-config', default='global.config.yml', help='Path of the global configuration file')
parser.add_argument('--data-main', default='data.main.pkl', help='Path of the main data file')
parser.add_argument('--seed', type=int, required=False, help='Seed for the generation of random values')
parser.add_argument('--cuda', action='store_true', help='Whether you want to run in GPU')
parser.add_argument('--cuda-device', default=0, type=int, help='GPU device to run into')
parser.add_argument('--parallel', action='store_true', help='Run in multiple GPUs if available')
parser.add_argument('--verbose', action='store_true', help='Whether to output the log using the standard output')

# Create subparsers
subparsers = parser.add_subparsers(dest='command')

# Create the subparser for the 'create_data' command
subparser_create_data = subparsers.add_parser(name='create_data_main')

# Create the subparser for the 'init' command
subparser_init = subparsers.add_parser(name='init')
subparser_init.add_argument('--exp-name', required=True, help='Name of the experiment to initialize')
subparser_init.add_argument('--split-config', required=True, help='Path to the split configuration to be used')
subparser_init.add_argument('--model-config', required=True, help='Path to the model configuration to be used')

# Create the subparser for the 'init_adaptation' command
subparser_init_adaptation = subparsers.add_parser(name='init_adaptation')
subparser_init_adaptation.add_argument('--exp-name', required=True, help='Name of the experiment to initialize')
subparser_init_adaptation.add_argument('--exp-base-name', required=True, help='Name of the experiment to initialize '
                                                                              'from')
subparser_init_adaptation.add_argument('--checkpoint-base-epoch', required=True, help='Base epoch to start the '
                                                                                      'adaptation')

# Create the subparser for the 'train' command
subparser_train = subparsers.add_parser('train')
subparser_train.add_argument('--exp-name', required=True, help='Name of the experiment to model')

# Create the subparser for the 'test' command
subparser_test = subparsers.add_parser('test')
subparser_test.add_argument('--exp-name', required=True, help='Name of the experiment to model')
subparser_test.add_argument('--checkpoint-epoch', type=int, nargs="*", required=False, default=0)

# Create the subparser for the 'test' command
subparser_test_speaker = subparsers.add_parser('test_speaker')
subparser_test_speaker.add_argument('--exp-name', required=True, help='Name of the experiment to model')
subparser_test_speaker.add_argument('--checkpoint-epoch', type=int, nargs="*", required=False, default=0)
subparser_test_speaker.add_argument('--speaker-id', type=str, nargs="*", required=True)
subparser_test_speaker.add_argument('--seed-duration', type=int, nargs="*", required=True, default=60)

# Create the subparser for the 'infer' command
subparser_infer = subparsers.add_parser('infer')
subparser_infer.add_argument('--exp-name', required=True, help='Name of the experiment to model')
subparser_infer.add_argument('--checkpoint-epoch', type=int, nargs="*", required=False, default=0)

# Get arguments
args = parser.parse_args()

# Create the exection object
execution = SampleRNNExecution(
    command=args.command,
    experiment_name=(args.exp_name if 'exp_name' in args else None),
    checkpoint_epoch=(args.checkpoint_epoch if 'checkpoint_epoch' in args else None),
    experiment_base_name=(args.exp_base_name if 'exp_base_name' in args else None),
    checkpoint_base_epoch=(args.checkpoint_base_epoch if 'checkpoint_base_epoch' in args else None),
    global_config=(args.global_config if 'global_config' in args else None),
    split_config=(args.split_config if 'split_config' in args else None),
    model_config=(args.model_config if 'model_config' in args else None),
    data_main_path=args.data_main,
    seed=(args.seed if 'seed' in args else None),
    cuda=args.cuda,
    cuda_device=args.cuda_device,
    parallel=args.parallel,
    verbose=args.verbose
)

# Actions to perform with the 'create_data' command
if args.command == 'create_data_main':
    conf = SampleRNNConfiguration(logger=execution.logger)
    conf.load_file(config_file_path=execution.global_config)
    data = SampleRNNData(conf=conf, logger=execution.logger)
    data.create_data_main()
    data.save_data_main(data_main_file_path=execution.data_main_path)

# Actions to perform with the 'init' command
elif args.command == 'init':
    execution.experiment.create()
    conf = SampleRNNConfiguration(logger=execution.logger)
    conf.load_file(config_file_path=execution.global_config)
    conf.load_file(config_file_path=execution.split_config)
    data = SampleRNNData(conf=conf, logger=execution.logger)
    data.load_data_main(data_main_file_path=execution.data_main_path)
    data.create_data_split()
    data.save_data_split(data_split_file_path=execution.experiment.data_file_path)
    execution.log_execution_entry()

# Actions to perform with the 'init' command
elif args.command == 'init_adaptation':
    execution.experiment_base.load()
    execution.experiment.create_adaptation()
    execution.log_execution_entry()

# Actions to perform with the 'train' command
elif args.command == 'train':
    execution.experiment.load()
    execution.log_execution_entry()
    SampleRNN(execution=execution).train()

# Actions to perform with the 'test' command
elif args.command == 'test':
    execution.experiment.load()
    execution.log_execution_entry()
    SampleRNN(execution=execution).test()

# Actions to perform with the 'test_speaker' command
elif args.command == 'test_speaker':
    execution.experiment.load()
    execution.log_execution_entry()
    SampleRNN(execution=execution).test_speaker_by_seed(speaker_id=args.speaker_id, seed_durations=args.seed_duration)

# Actions to perform with the 'infer' command
elif args.command == 'infer':
    execution.experiment.load()
    execution.log_execution_entry()
    SampleRNN(execution=execution).infer(n_samples=5)

