# SampleRNN in PyTorch

Code used in [Problem-Agnostic Speech Embeddings for Multi-Speaker Text-to-Speech with SampleRNN](https://arxiv.org/abs/1906.00733)


## Dependencies

The dependencies of the repository are specified using the standard 
`requirements.txt` file. To install them, just execute the following command 
using the desired `pip binary:

```pip install -r requirements.txt```

The code has been tested using _Python 3.7_.

## Executing the code

The code must be executed calling the package directly. There four different 
commands, which should be executed sequentially:

````
Usage:
    python -m samplernn [--global-config <global_config_file_path>] [--seed <seed>] [--cuda] [--cuda-device 
    <cuda_device>] [--parallel] [--verbose] <COMMAND>
    
Options:
    --global-config <global_config_file_path>           Full path to global.config.yml file.
    --seed <seed>                                       SEED to be used in the initialization.
    --cuda                                              Whether or not to run in CUDA.
    --cuda-device <cuda_device>                         CUDA device to use.
    --parallel                                          Whether or not to run in multiple GPUs.
    --verbose                                           Whether or not to log using standard output.
    
Commands:
    init --exp-name <experiment_name> --split-config <split_config_file_path> --model-config <model_config_file_path>
    init_adaptation --exp-name <experiment_name> [--exp-base-name <base_experiment_name>] [--checkpoint-base-epoch 
    <base_experiment_epoch>]
    train --exp-name <experiment_name>
    test --exp-name <experiment_name> --checkpoint-epoch <checkpoint_epoch>
    test_speaker --exp-name <experiment_name> --checkpoint-epoch <checkpoint_epoch> --speaker-id <speaker_ids> 
    --seed-duration <seed_duration>
    infer --exp-name <experiment_name> --checkpoint-epoch <checkpoint_epoch>
      
Commands Options:
    --exp-name <experiment_name>                        Name of the experiment of the execution.
    --split-config <split_config_file_path>             Full path to split.config.yml file.
    --model-config <model_config_file_path>             Full path to model.config.yml file.
    --exp-base-name <base_experiment_name>              Name of the experiment to use as a base for adaptation.
    --checkpoint-base-epoch <base_experiment_epoch>     Checkpoint to use as initial state of the model from the base
                                                        experiment.
    --checkpoint-epoch <checkpoint_epoch>               Epoch to perform the tests.
    --speaker-id <speaker_ids>                          List of SpeakerIDs to test
    --seed-duration <seed_duration>                     List of PASE seed durations to test
````

## References

* [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837)
* [SampleRNN implementation in PyTorch by DeepSound](https://github.com/deepsound-project/samplernn-pytorch)