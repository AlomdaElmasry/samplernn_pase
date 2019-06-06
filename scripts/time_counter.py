from samplernn.execution import SampleRNNExecution
from samplernn.experiment import SampleRNNExperiment
from samplernn.loader import SampleRNNDataLoader
from samplernn.model import SampleRNNModel
from samplernn.utils import SampleRNNQuantizer
import pase.frontend
import time

start = time.time()

execution = SampleRNNExecution(
    command='train',
    experiment_name='40spk_linglf0_pase_trained',
    checkpoint_epoch=None,
    experiment_base_name=None,
    checkpoint_base_epoch=None,
    global_config='/Users/DavidAlvarezDLT/PycharmProjects/samplernn_pytorch/global.config.yml',
    split_config=None,
    model_config=None,
    data_main_path='/Users/DavidAlvarezDLT/PycharmProjects/samplernn_pytorch/data.main.pkl',
    seed=1234,
    cuda=False,
    cuda_device=0,
    parallel=False,
    verbose=True
)
execution.experiment.load()
quantizer = SampleRNNQuantizer(conf=execution.experiment.conf)
train_data_loader = SampleRNNDataLoader(
    execution=execution,
    quantizer=quantizer,
    is_adaptation=execution.experiment.is_adaptation,
    split='validation'
)

loading_time = time.time()
print('Loading Experiment: {}'.format(loading_time-start))

model = SampleRNNModel(
    conf=execution.experiment.conf,
    quantizer=quantizer,
    conds_linguistic_n=[
        len(execution.experiment.data.utterances_conds_linguistic_categories['phonemes']),
        len(execution.experiment.data.utterances_conds_linguistic_categories['vowels']),
        len(execution.experiment.data.utterances_conds_linguistic_categories['gpos']),
        len(execution.experiment.data.utterances_conds_linguistic_categories['tobi'])
    ]
)
model = model

load_samplernn = time.time()
print('Loading SampleRNN: {}'.format(load_samplernn-loading_time))

pase_encoder = pase.frontend.wf_builder(execution.experiment.conf.pase['config_file_path'])
pase_encoder.load_pretrained(execution.experiment.conf.pase['trained_model_path'], load_last=True, verbose=True)
pase_encoder = pase_encoder

load_pase = time.time()
print('Loading PASE: {}'.format(load_pase-load_samplernn))
start = 0
for i, data in enumerate(train_data_loader):
    init_it = time.time()
    if i > 0:
        print('Load Data Iteration {}: {}'.format(i, init_it-end_it))
    data_samples, data_samples_target, data_conds_speakers, data_conds_utterances, data_model_reset, data_info = \
        data
    speaker_indexes = [data_info_item['speaker'] if data_info_item is not None else None for data_info_item in
                       data_info]
    pase_chunks = train_data_loader.get_random_chunks(speaker_indexes, 16000).unsqueeze(1)
    pase_chunks = pase_chunks
    prepare_data = time.time()
    print('Prepare PASE data Iteration {}: {}'.format(i, prepare_data-init_it))
    pase_output = pase_encoder(pase_chunks)
    forward_pase = time.time()
    print('Forward PASE Iteration {}: {}'.format(i, forward_pase - prepare_data))
    end_it = time.time()
