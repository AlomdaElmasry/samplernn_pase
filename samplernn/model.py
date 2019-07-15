import numpy as np
import torch
from .configuration import SampleRNNConfiguration
from .utils import SampleRNNQuantizer, lecun_uniform, concat_init
from typing import Dict


class FrameLevelSampleRNNModel(torch.nn.Module):
    """Frame level module of the SampleRNN architecture"""

    frame_input_samples: int
    frame_ratio: int
    rnn_layers: int
    rnn_hidden_size: int
    conds_size: int

    _samples_expand_layer: torch.nn.Conv1d
    _conds_expand_layer: torch.nn.Conv1d

    _rnn_layer: torch.nn.GRU
    _rnn_layer_h0: torch.nn.Parameter

    _upsampling_layer: torch.nn.ConvTranspose1d
    _upsampling_layer_bias: torch.nn.Parameter

    def __init__(self, frame_input_samples: int, frame_ratio: int, rnn_layers: int, rnn_hidden_size: int,
                 conds_size: int):

        # Call parent constructor
        super().__init__()

        # Store class parameters
        self.frame_input_samples = frame_input_samples
        self.frame_ratio = frame_ratio
        self.rnn_layers = rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.conds_size = conds_size

        # Create self._samples_expand_layer
        self._samples_expand_layer = torch.nn.Conv1d(
            in_channels=frame_input_samples,
            out_channels=rnn_hidden_size,
            kernel_size=1
        )

        # Create self._conds_expand_layer
        self._conds_expand_layer = torch.nn.Conv1d(
            in_channels=conds_size,
            out_channels=rnn_hidden_size,
            kernel_size=1
        )

        # Create self._rnn_layer
        self._rnn_layer = torch.nn.GRU(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True
        )

        # Create self._rnn_layer_h0
        self._rnn_layer_h0 = torch.nn.Parameter(torch.zeros(rnn_layers, rnn_hidden_size))

        # Create self._upsampling_layer
        self._upsampling_layer = torch.nn.ConvTranspose1d(
            in_channels=rnn_hidden_size,
            out_channels=rnn_hidden_size,
            kernel_size=frame_ratio,
            stride=frame_ratio,
            bias=False
        )

        # Create self._upsampling_layer_bias
        self._upsampling_layer_bias = torch.nn.Parameter(torch.FloatTensor(rnn_hidden_size, frame_ratio))

        # Reset Parameters
        self._upsampling_layer.reset_parameters()

        # Initialize learnable parameters
        self._initialize_learnable_parameters()
        self._normalize_learnable_parameters()

    def _initialize_learnable_parameters(self):
        """Initializes the learnable parameters of the SampleLevelSampleRNNModel module"""

        torch.nn.init.kaiming_uniform_(self._samples_expand_layer.weight)
        torch.nn.init.constant_(self._samples_expand_layer.bias, 0)

        if self.conds_size is not None:
            torch.nn.init.kaiming_uniform_(self._conds_expand_layer.weight)
            torch.nn.init.constant_(self._conds_expand_layer.bias, 0)

        torch.nn.init.uniform_(
            self._upsampling_layer.weight,
            -np.sqrt(6 / self.rnn_hidden_size),
            np.sqrt(6 / self.rnn_hidden_size)
        )

        torch.nn.init.constant_(self._upsampling_layer_bias, 0)

        for i in range(self.rnn_layers):
            concat_init(
                getattr(self._rnn_layer, 'weight_ih_l{}'.format(i)),
                [lecun_uniform, lecun_uniform, lecun_uniform]
            )
            torch.nn.init.constant_(getattr(self._rnn_layer, 'bias_ih_l{}'.format(i)), 0)
            concat_init(
                getattr(self._rnn_layer, 'weight_hh_l{}'.format(i)),
                [lecun_uniform, lecun_uniform, torch.nn.init.orthogonal_]
            )
            torch.nn.init.constant_(getattr(self._rnn_layer, 'bias_hh_l{}'.format(i)), 0)

    def _normalize_learnable_parameters(self):
        """Normalizes the learnable parameters of the SampleLevelSampleRNNModel module"""

        self._samples_expand_layer = torch.nn.utils.weight_norm(self._samples_expand_layer)

        if self.conds_size is not None:
            self._conds_expand_layer = torch.nn.utils.weight_norm(self._conds_expand_layer)

        self._upsampling_layer = torch.nn.utils.weight_norm(self._upsampling_layer)

    def forward(self, input_samples, input_conds, upper_tier_conditioning, rnn_hidden_state):
        """FrameLevelSampleRNNModel forwarding function of the SampleRNN architecture

        Args:
            input_samples (torch.Tensor): matrix of (batch_size, sequence_length, frame_input_samples) containing the sample
            inputs of the sample level module
            upper_tier_conditioning (torch.Tensor): matrix of (batch_size, sequence_length * (prev) frame_ratio, rnn_hidden_size)
            rnn_hidden_state (torch.Tensor): matrix of (rnn_layers, batch_size, rnn_hidden_size)

        Returns:
            upsampling_output (torch.Tensor): matrix of (batch_size, sequence_length * frame_ratio, rnn_hidden_size)
            rnn_hidden_state_new (torch.Tensor): matrix of (rnn_layers, batch_size, rnn_hidden_size)
        """

        # Obtain the batch size
        (batch_size, sequence_length, _) = input_samples.size()

        # Check if we have to upscale the conds
        if sequence_length != input_conds.shape[1]:
            upscale_ratio = int(input_samples.shape[1] / input_conds.shape[1])
            input_conds = input_conds.unsqueeze(2) \
                .expand(batch_size, input_conds.shape[1], upscale_ratio, input_conds.shape[2]) \
                .reshape(batch_size, sequence_length, input_conds.shape[2])

        # samples_expand_output is (batch_size, sequence_length, rnn_hidden_size)
        samples_expand_output = self._samples_expand_layer(input_samples.permute(0, 2, 1)).permute(0, 2, 1)
        conds_expand_output = self._conds_expand_layer(input_conds.permute(0, 2, 1)).permute(0, 2, 1)

        # Check if the conds are available
        samples_expand_output += conds_expand_output

        # Add conditioning if exists
        if upper_tier_conditioning is not None:
            samples_expand_output += upper_tier_conditioning

        # Initialize hidden state tensor
        hidden_state_tensor = torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size)

        # Move it to CUDA, if available
        if input_samples.is_cuda:
            hidden_state_tensor = hidden_state_tensor.cuda()

        # Iterate over hidden state list
        for hidden_state_item_index, hidden_state_item in enumerate(rnn_hidden_state):

            # If the item is None, initialize it
            if hidden_state_item is None:
                hidden_state_tensor[:, hidden_state_item_index, :] = self._rnn_layer_h0.unsqueeze(1)

            # If the item is not None, assign it
            else:
                hidden_state_tensor[:, hidden_state_item_index, :] = hidden_state_item.unsqueeze(1)

        # rnn_output is (batch_size, sequence_length, rnn_hidden_size)
        # rnn_hidden_state_new is (rnn_layers, batch_size, rnn_hidden_size)
        (rnn_output, rnn_hidden_state_new) = self._rnn_layer(samples_expand_output, hidden_state_tensor)

        # upsampling_bias is (batch_size, self.rnn_hidden_size, sequence_length * self.frame_ratio)
        upsampling_bias = self._upsampling_layer_bias.unsqueeze(0).unsqueeze(2) \
            .expand(batch_size, self.rnn_hidden_size, sequence_length, self.frame_ratio) \
            .contiguous().view(batch_size, self.rnn_hidden_size, sequence_length * self.frame_ratio)

        # upsampling_output is (batch_size, sequence_length * frame_ratio, rnn_hidden_size)
        upsampling_output = (self._upsampling_layer(rnn_output.permute(0, 2, 1)) + upsampling_bias).permute(0, 2, 1)

        # Return the output and the new hidden state
        return upsampling_output, rnn_hidden_state_new


class SampleLevelSampleRNNModel(torch.nn.Module):
    """Sample level module of the SampleRNN architecture"""

    # Integer containining the number of samples entering the sample level module
    frame_input_samples: int
    conds_size: int
    rnn_hidden_size: int
    q_levels: int

    # Embedding layer used to transform from (batch_size, 1059) to (batch_size, 1059, embedding_dim)
    _embedding_layer: torch.nn.Embedding

    _embedding_expand_layer: torch.nn.Conv1d
    _conds_expand_layer: torch.nn.Conv1d

    _inputs_comb_layer: torch.nn.Linear

    _global_expand_layer: torch.nn.Conv1d
    _adaptation_layer: torch.nn.Conv1d
    _logsoftmax_layer: torch.nn.LogSoftmax

    def __init__(self, frame_input_samples: int, conds_size: int, rnn_hidden_size: int, q_levels: int):
        # Call parent constructor
        super().__init__()

        # Store class parameters
        self.frame_input_samples = frame_input_samples
        self.conds_size = conds_size
        self.rnn_hidden_size = rnn_hidden_size
        self.q_levels = q_levels

        # Create Torch objects
        self._embedding_layer = torch.nn.Embedding(num_embeddings=q_levels, embedding_dim=q_levels)

        # lala
        self._embedding_expand_layer = torch.nn.Conv1d(
            in_channels=q_levels,
            out_channels=rnn_hidden_size,
            kernel_size=frame_input_samples, bias=False
        )
        self._conds_expand_layer = torch.nn.Conv1d(
            in_channels=conds_size,
            out_channels=rnn_hidden_size,
            kernel_size=1
        )

        # Lele
        self._inputs_comb_layer = torch.nn.Linear(
            in_features=self.rnn_hidden_size * 3,
            out_features=self.rnn_hidden_size
        )

        # Lolo
        self._global_expand_layer = torch.nn.Conv1d(
            in_channels=rnn_hidden_size,
            out_channels=rnn_hidden_size,
            kernel_size=1
        )

        # Lulu
        self._adaptation_layer = torch.nn.Conv1d(
            in_channels=rnn_hidden_size,
            out_channels=q_levels,
            kernel_size=1
        )

        # Lele
        self._softmax_layer = torch.nn.LogSoftmax(dim=2)

        # Initialize learnable parameters
        self._initialize_learnable_parameters()
        self._normalize_learnable_parameters()

    def _initialize_learnable_parameters(self):
        """Initializes the learnable parameters of the SampleLevelSampleRNNModel module"""

        torch.nn.init.kaiming_uniform_(self._embedding_expand_layer.weight)

        torch.nn.init.kaiming_uniform_(self._global_expand_layer.weight)
        torch.nn.init.constant_(self._global_expand_layer.bias, 0)

        lecun_uniform(self._adaptation_layer.weight)
        torch.nn.init.constant_(self._adaptation_layer.bias, 0)

    def _normalize_learnable_parameters(self):
        """Normalizes the learnable parameters of the SampleLevelSampleRNNModel module"""

        self._embedding_expand_layer = torch.nn.utils.weight_norm(self._embedding_expand_layer)
        self._global_expand_layer = torch.nn.utils.weight_norm(self._global_expand_layer)
        self._adaptation_layer = torch.nn.utils.weight_norm(self._adaptation_layer)

    def forward(self, input_samples, input_conds, upper_tier_conditioning):
        """SampleLevelSampleRNNModel forwarding function of the SampleRNN architecture

        Args:
            input_samples (torch.Tensor): matrix of (batch_size, 1059) containing the sample inputs of the sample
            level module
            upper_tier_conditioning (torch.Tensor): matrix of (batch_size, sequence_length * frame_size, rnn_hidden_size)

        Returns:

        """
        # Obtain the batch size
        batch_size, _ = input_samples.size()

        # Upscale the Conds
        upscale_ratio = int(upper_tier_conditioning.shape[1] / input_conds.shape[1])
        input_conds = input_conds.unsqueeze(2) \
            .expand(batch_size, input_conds.shape[1], upscale_ratio, input_conds.shape[2]) \
            .reshape(batch_size, upper_tier_conditioning.shape[1], input_conds.shape[2])

        # embedding_output is ()
        embedding_output = self._embedding_layer(input_samples.contiguous().view(-1)) \
            .view(batch_size, -1, self.q_levels)

        # Expand both Samples and Conds
        embedding_expand_output = self._embedding_expand_layer(embedding_output.permute(0, 2, 1))
        conds_expand_output = self._conds_expand_layer(input_conds.permute(0, 2, 1))

        # Apply Fully-Connected to Samples, Conds and UpperTier
        inputs_comb_output = self._inputs_comb_layer(torch.cat(
            (embedding_expand_output.permute(0, 2, 1), conds_expand_output.permute(0, 2, 1), upper_tier_conditioning),
            dim=2)
        )
        inputs_comb_output = torch.nn.functional.relu(inputs_comb_output)

        # global_expand_output is ()
        global_expand_output = self._global_expand_layer(inputs_comb_output.permute(0, 2, 1))
        global_expand_output = torch.nn.functional.relu(global_expand_output)

        # adaptation_output is ()
        adaptation_output = self._adaptation_layer(global_expand_output)

        # Apply the LogSoftMax layer and return the result as (batch_size, sequence_length * frame_size, ,q_levels)
        return self._softmax_layer(adaptation_output.permute(0, 2, 1))


class SampleRNNModel(torch.nn.Module):
    """General module of the SampleRNN architecture"""

    # Lala
    conf: SampleRNNConfiguration
    quantizer: SampleRNNQuantizer

    conds_linguistic_phonemes: torch.nn.Embedding
    conds_linguistic_vowels: torch.nn.Embedding
    conds_linguistic_gpos: torch.nn.Embedding
    conds_linguistic_tobi: torch.nn.Embedding

    _conds_adaptation_layer: torch.nn.Linear

    frame_level_layers: torch.nn.ModuleList
    sample_level_layer: SampleLevelSampleRNNModel

    frame_level_hidden_states: Dict

    def __init__(self, conf: SampleRNNConfiguration, quantizer: SampleRNNQuantizer, conds_linguistic_n=None):

        # Call parent constructor
        super().__init__()

        # Store class parameters
        self.conf = conf
        self.quantizer = quantizer

        # Initialize parameters for FrameLevelLayers
        self.frame_level_layers = torch.nn.ModuleList()
        self.conds_linguistic_phonemes = torch.nn.Embedding(
            num_embeddings=conds_linguistic_n[0],
            embedding_dim=self.conf.conditionants['utterance_linguistic_phonemes_embedding_size']
        )
        self.conds_linguistic_vowels = torch.nn.Embedding(
            num_embeddings=conds_linguistic_n[1],
            embedding_dim=self.conf.conditionants['utterance_linguistic_vowels_embedding_size']
        )
        self.conds_linguistic_gpos = torch.nn.Embedding(
            num_embeddings=conds_linguistic_n[2],
            embedding_dim=self.conf.conditionants['utterance_linguistic_gpos_embedding_size']
        )
        self.conds_linguistic_tobi = torch.nn.Embedding(
            num_embeddings=conds_linguistic_n[3],
            embedding_dim=self.conf.conditionants['utterance_linguistic_tobi_embedding_size']
        )

        # Create the Conds Adaptation Layer
        self._conds_adaptation_layer = torch.nn.Linear(
            in_features=self.conf.conditionants['utterance_size_expanded'] + self.conf.conditionants['speaker_size'],
            out_features=self.conf.conditionants['global_size']
        )

        # Lala
        for layer_n in range(0, len(self.conf.architecture['frame_layers_fs'])):
            self.frame_level_layers.append(
                FrameLevelSampleRNNModel(
                    frame_input_samples=self.conf.architecture['frame_layers_fs'][layer_n],
                    frame_ratio=self.conf.architecture['frame_layers_ratios'][layer_n],
                    rnn_layers=self.conf.architecture['frame_layers_rnn_layers'][layer_n],
                    rnn_hidden_size=self.conf.architecture['frame_layers_rnn_hidden_size'][layer_n],
                    conds_size=self.conf.conditionants['global_size']
                )
            )

        # Initialize SampleLevelRNN
        self.sample_level_layer = SampleLevelSampleRNNModel(
            frame_input_samples=conf.architecture['frame_layers_ratios'][0],
            conds_size=self.conf.conditionants['global_size'],
            rnn_hidden_size=conf.architecture['frame_layers_rnn_hidden_size'][0],
            q_levels=conf.quantizer['q_levels']
        )

        # Initialize Hidden States
        self.frame_level_hidden_states = None

    def _get_frame_level_hidden_states(self, frame_level_layer, reset_list):

        # Define returned Tensor
        frame_level_layer_hidden_state = []

        # Iterate over the batch_size elements
        for reset_index, reset_element in enumerate(reset_list):

            # If the element is False, get stored item
            if reset_element == 0:
                frame_level_layer_hidden_state.append(self.frame_level_hidden_states[frame_level_layer][reset_index])

            # If the element is True, set None to that element
            elif reset_element == 1:
                frame_level_layer_hidden_state.append(None)

        # Return the list
        return frame_level_layer_hidden_state

    def _set_frame_level_hidden_states(self, new_hidden_state_tensor, frame_level_layer: FrameLevelSampleRNNModel,
                                       reset_list):

        # Create aux var
        last_hidden_state = 0

        # Iterate over the batch_size elements
        for reset_index, reset_element in enumerate(reset_list):

            # Assign only if reset_element == 1 or 0
            if reset_element == 0 or reset_element == 1:
                self.frame_level_hidden_states[frame_level_layer][reset_index] = \
                    new_hidden_state_tensor[:, last_hidden_state, :]
                last_hidden_state += 1
            else:
                self.frame_level_hidden_states[frame_level_layer][reset_index] = None

    def _format_linguistic_features(self, input_conds):
        # Create aux conds Tensor
        input_conds_aux = torch.zeros(
            (input_conds.shape[0], input_conds.shape[1], self.conf.conditionants['utterance_size_expanded'])
        )

        # Shorcuts for embedding sizes
        phonemes_size = self.conf.conditionants['utterance_linguistic_phonemes_embedding_size']
        vowels_size = self.conf.conditionants['utterance_linguistic_vowels_embedding_size']
        gpos_size = self.conf.conditionants['utterance_linguistic_gpos_embedding_size']
        tobi_size = self.conf.conditionants['utterance_linguistic_tobi_embedding_size']

        # Define aux variable
        last_index = 0

        # Append CATEGORICAL features at the beginning
        input_conds_aux[:, :, last_index:last_index + phonemes_size] = self.conds_linguistic_phonemes(
            input_conds[:, :, 2].long())
        last_index += phonemes_size

        input_conds_aux[:, :, last_index:last_index + phonemes_size] = self.conds_linguistic_phonemes(
            input_conds[:, :, 3].long())
        last_index += phonemes_size

        input_conds_aux[:, :, last_index:last_index + phonemes_size] = self.conds_linguistic_phonemes(
            input_conds[:, :, 4].long())
        last_index += phonemes_size

        input_conds_aux[:, :, last_index:last_index + phonemes_size] = self.conds_linguistic_phonemes(
            input_conds[:, :, 5].long())
        last_index += phonemes_size

        input_conds_aux[:, :, last_index:last_index + phonemes_size] = self.conds_linguistic_phonemes(
            input_conds[:, :, 6].long())
        last_index += phonemes_size

        input_conds_aux[:, :, last_index:last_index + vowels_size] = self.conds_linguistic_vowels(
            input_conds[:, :, 27].long())
        last_index += vowels_size

        input_conds_aux[:, :, last_index:last_index + gpos_size] = self.conds_linguistic_gpos(
            input_conds[:, :, 31].long())
        last_index += gpos_size

        input_conds_aux[:, :, last_index:last_index + gpos_size] = self.conds_linguistic_gpos(
            input_conds[:, :, 33].long())
        last_index += gpos_size

        input_conds_aux[:, :, last_index:last_index + gpos_size] = self.conds_linguistic_gpos(
            input_conds[:, :, 41].long())
        last_index += gpos_size

        input_conds_aux[:, :, last_index:last_index + tobi_size] = self.conds_linguistic_tobi(
            input_conds[:, :, 49].long())
        last_index += tobi_size

        # Append REAL and BOOL features after the embeddings
        input_conds_aux[:, :, last_index:last_index + 2] = input_conds[:, :, 0:2]
        last_index += 2

        input_conds_aux[:, :, last_index:last_index + 20] = input_conds[:, :, 7:27]
        last_index += 20

        input_conds_aux[:, :, last_index:last_index + 3] = input_conds[:, :, 28:31]
        last_index += 3

        input_conds_aux[:, :, last_index:last_index + 1] = input_conds[:, :, 32:33]
        last_index += 1

        input_conds_aux[:, :, last_index:last_index + 7] = input_conds[:, :, 34:41]
        last_index += 7

        input_conds_aux[:, :, last_index:last_index + 7] = input_conds[:, :, 42:49]
        last_index += 7

        input_conds_aux[:, :, last_index:] = input_conds[:, :, 50:]

        # Move to CUDA if required
        if input_conds.is_cuda:
            input_conds_aux = input_conds_aux.cuda()

        # Return it
        return input_conds_aux

    def forward(self, utterance_samples, speaker_conds, utterance_conds, utterances_reset):

        # Get basic Parameters
        batch_size, time_steps, _ = utterance_conds.shape

        # Initialize Hidden States Dict
        if self.frame_level_hidden_states is None:
            self.frame_level_hidden_states = {
                rnn: [None] * utterance_conds.shape[0] for rnn in self.frame_level_layers
            }

        # Check that there are valid samples to propagate
        if not any(utterances_reset != 2):
            return_tensor = torch.zeros(
                utterance_samples.shape[0],
                self.conf.architecture['receptive_field'],
                self.quantizer.q_levels
            )
            if utterance_samples.is_cuda:
                return_tensor = return_tensor.cuda()
            return return_tensor

        # Clean the inputs
        else:
            utterance_samples = utterance_samples[utterances_reset != 2] if utterance_samples is not None else None
            utterance_conds = utterance_conds[utterances_reset != 2]
            speaker_conds = speaker_conds[utterances_reset != 2]

        # Check if we are dealing with linguistic conditionants to apply the embeddings
        if self.conf.conditionants['utterance_type'] in ['linguistic', 'linguistic_lf0']:
            utterance_conds = self._format_linguistic_features(utterance_conds)

        # Prepare Conds
        speaker_conds = speaker_conds.unsqueeze(1).expand(utterance_conds.shape[0], time_steps, -1)

        # Apply Linear transformation to the input conds
        input_conds = self._conds_adaptation_layer(torch.cat((utterance_conds, speaker_conds), dim=2))

        # Training Mode
        if self.training:

            # Create holder of the result
            return_tensor = torch.zeros(batch_size, self.conf.architecture['receptive_field'], self.quantizer.q_levels)

            # Move to CUDA if required
            if utterance_samples.is_cuda:
                return_tensor = return_tensor.cuda()

            # Get the model output
            model_output = self.do_train(
                input_samples=utterance_samples,
                input_conds=input_conds,
                utterances_reset=utterances_reset
            )

            # Store the result in the appropiate positions
            last_index = 0
            for reset_index, reset_item in enumerate(utterances_reset):
                if reset_item != 2:
                    return_tensor[reset_index, :, :] = model_output[last_index, :, :]
                    last_index += 1

            # Return the torch.Tensor
            return return_tensor

        # Inference Mode
        else:

            # Create holder of the result
            return_tensor = torch.zeros(batch_size, self.conf.architecture['frame_size'] + time_steps *
                                        self.conf.architecture['frame_size'])

            # Move to CUDA if required
            if utterance_conds.is_cuda:
                return_tensor = return_tensor.cuda()

            # Get the model output
            model_output = self.do_infer(
                utterances_conds=input_conds,
                utterances_reset=utterances_reset
            )

            # Store the result in the appropiate positions
            last_index = 0
            for reset_index, reset_item in enumerate(utterances_reset):
                if reset_item != 2:
                    return_tensor[reset_index, :] = model_output[last_index, :]
                    last_index += 1

            # Return the torch.Tensor
            return return_tensor

    def do_train(self, input_samples, input_conds, utterances_reset):

        # Get batch_size
        (batch_size, _) = input_samples.size()

        # Initialize upper level conditioners
        upper_tier_conditioning = None

        # Iterate over the list of sample level layers
        for frame_level_layer in reversed(self.frame_level_layers):
            # Compute samples to pass in current frame level layer
            from_index = self.frame_level_layers[-1].frame_input_samples - frame_level_layer.frame_input_samples
            to_index = -frame_level_layer.frame_input_samples + 1

            # Quantize the samples
            frame_layer_input_samples = self.quantizer.dequantize(input_samples[:, from_index: to_index])

            # Reshape samples to (batch_size, seq_len, frame_level_fs)
            frame_layer_input_samples = frame_layer_input_samples.contiguous() \
                .view(batch_size, -1, frame_level_layer.frame_input_samples)

            # Get next frame level hidden state
            frame_level_hidden_state = self._get_frame_level_hidden_states(
                frame_level_layer=frame_level_layer,
                reset_list=utterances_reset
            )

            # Propagate through current frame level layer
            (upper_tier_conditioning, new_hidden) = frame_level_layer(
                input_samples=frame_layer_input_samples,
                input_conds=input_conds,
                upper_tier_conditioning=upper_tier_conditioning,
                rnn_hidden_state=frame_level_hidden_state
            )

            # Store new hidden state in the dictionary
            self._set_frame_level_hidden_states(new_hidden.detach(), frame_level_layer, utterances_reset)

        # Get sample level input
        sample_layer_input_samples = input_samples[:, (self.frame_level_layers[-1].frame_input_samples -
                                                       self.sample_level_layer.frame_input_samples):]

        # Propagate through sample level layer and return the result
        return self.sample_level_layer(
            input_samples=sample_layer_input_samples,
            input_conds=input_conds,
            upper_tier_conditioning=upper_tier_conditioning
        )

    def do_infer(self, utterances_conds, utterances_reset):
        # Get batch_size
        (batch_size, num_portions, conds_size) = utterances_conds.size()

        # Create a Tensor to store the generated samples in
        generated_sequences = torch.zeros(
            batch_size,
            self.conf.architecture['frame_size'] + num_portions * self.conf.architecture['frame_size'],
            dtype=torch.int64
        ).fill_(self.quantizer.quantize_zero())

        # Move to CUDA
        if utterances_conds.is_cuda:
            generated_sequences = generated_sequences.cuda()

        # Create a list to store the conditioning
        frame_level_outputs = [None for _ in self.frame_level_layers]

        # Iterate over the samples
        for generated_sample in range(self.conf.architecture['frame_size'], generated_sequences.shape[1]):
            # Compute conds index
            conds_indx, _ = divmod(generated_sample, self.conf.architecture['frame_size'])
            conds_indx -= 1

            # On
            if generated_sample == self.conf.architecture['frame_size'] + 1:
                utterances_reset[utterances_reset == 1] = 0

            # Iterate over Frame Level layers
            for (frame_level_indx, frame_level_layer) in reversed(list(enumerate(self.frame_level_layers))):

                # If the generated sample is not a multiple of the input size, skip
                if generated_sample % frame_level_layer.frame_input_samples != 0:
                    continue

                # Prepare the input samples to enter the model
                frame_layer_input_samples = torch.autograd.Variable(self.quantizer.dequantize(
                    generated_sequences[:, generated_sample - frame_level_layer.frame_input_samples:generated_sample]
                ).unsqueeze(1))

                # Mode the variable to CUDA, if available
                if utterances_conds.is_cuda:
                    frame_layer_input_samples = frame_layer_input_samples.cuda()

                # Check if we have conditioning
                if frame_level_indx == len(self.frame_level_layers) - 1:
                    upper_tier_conditioning = None

                # If we are not in the last tier
                else:

                    # Compute frame_index
                    frame_index = (generated_sample // frame_level_layer.frame_input_samples) % \
                                  self.frame_level_layers[frame_level_indx + 1].frame_ratio

                    # Get the upper tier conditioning from the previous upper tier
                    upper_tier_conditioning = frame_level_outputs[frame_level_indx + 1][:, frame_index, :] \
                        .unsqueeze(1)

                # Set the new hidden states
                frame_level_hidden_state = self._get_frame_level_hidden_states(
                    frame_level_layer=frame_level_layer,
                    reset_list=utterances_reset
                )

                # Propagate through current frame level layer
                frame_level_outputs[frame_level_indx], new_frame_level_hiddden_state = \
                    frame_level_layer(
                        input_samples=frame_layer_input_samples,
                        input_conds=utterances_conds[:, conds_indx, :].unsqueeze(1),
                        upper_tier_conditioning=upper_tier_conditioning,
                        rnn_hidden_state=frame_level_hidden_state
                    )

                # Set the new frame level hidden state
                self._set_frame_level_hidden_states(
                    new_hidden_state_tensor=new_frame_level_hiddden_state.detach(),
                    frame_level_layer=frame_level_layer,
                    reset_list=utterances_reset
                )

            # Prepare the input samples Sample Level Layer
            sample_layer_input_samples = \
                generated_sequences[:, generated_sample - self.sample_level_layer.frame_input_samples:generated_sample]

            # Mode the variable to CUDA, if available
            if utterances_conds.is_cuda:
                sample_layer_input_samples = sample_layer_input_samples.cuda()

            # Prepare conditioning
            upper_tier_conditioning = frame_level_outputs[0][:, generated_sample % self.sample_level_layer
                .frame_input_samples, :].unsqueeze(1)

            # Store generated samples
            generated_sequences[:, generated_sample] = self.sample_level_layer(
                input_samples=sample_layer_input_samples,
                input_conds=utterances_conds[:, conds_indx, :].unsqueeze(1),
                upper_tier_conditioning=upper_tier_conditioning
            ).squeeze(1).exp_().multinomial(1).squeeze(1)

        # Return generated samples
        return generated_sequences
