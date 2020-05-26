import numpy as np
import samplernn_pase.utils as utils
import torch
import torch.nn.functional as F


class CondsMixer(torch.nn.Module):
    conds_speaker_type = None
    conds_speaker_size = None
    conds_utterance_type = None
    conds_utterance_size = None
    conds_utterance_expanded_size = None

    def __init__(self, conds_speaker_type, conds_speaker_n, conds_speaker_size, conds_utterance_type,
                 conds_utterance_linguistic_n, conds_utterance_linguistic_emb_size, conds_size):
        super(CondsMixer, self).__init__()
        self.conds_speaker_type = conds_speaker_type
        self.conds_utterance_type = conds_utterance_type
        self._init_dynamic_params(conds_utterance_linguistic_emb_size)
        self.speaker_embedding = torch.nn.Embedding(conds_speaker_n, conds_speaker_size)
        if self.conds_utterance_type in ['linguistic', 'linguistic_lf0']:
            emb_size = conds_utterance_linguistic_emb_size
            self.conds_utt_phonemes_emb = torch.nn.Embedding(conds_utterance_linguistic_n[0], emb_size)
            self.conds_utt_vowels_emb = torch.nn.Embedding(conds_utterance_linguistic_n[1], emb_size)
            self.conds_utt_gpos_emb = torch.nn.Embedding(conds_utterance_linguistic_n[2], emb_size)
            self.conds_utt_tobi_emb = torch.nn.Embedding(conds_utterance_linguistic_n[3], emb_size)
        self.conds_mix = torch.nn.Linear(self.conds_utterance_expanded_size + conds_speaker_size, conds_size)

    def _init_dynamic_params(self, conds_utterance_linguistic_emb_size):
        if self.conds_utterance_type == 'acoustic':
            self.conds_utterance_size = self.conds_utterance_expanded_size = 43
        elif self.conds_utterance_type == 'linguistic':
            self.conds_utterance_size = 55
            self.conds_utterance_expanded_size = 55 - 10 + 10 * conds_utterance_linguistic_emb_size
        elif self.conds_utterance_type == 'linguistic_lf0':
            self.conds_utterance_size = 57
            self.conds_utterance_expanded_size = 57 - 10 + 10 * conds_utterance_linguistic_emb_size

    def forward(self, utt_conds, info):
        speaker_conds = self._forward_speaker_conds(info, utt_conds.device).expand(
            utt_conds.size(0), utt_conds.size(1), -1
        )
        utt_conds = self._forward_linguistic_features(utt_conds)
        return self.conds_mix(torch.cat((speaker_conds, utt_conds), dim=2))

    def _forward_speaker_conds(self, info, device):
        if self.conds_speaker_type == 'embedding':
            speakers_ids = torch.tensor(
                [info_item['speaker']['index'] if info_item is not None else 0 for info_item in info], dtype=torch.int64
            ).to(device)
            return self.speaker_embedding(speakers_ids).unsqueeze(1)
        elif self.conds_speaker_type == 'pase':
            raise NotImplemented

    def _forward_linguistic_features(self, utt_conds):
        if self.conds_utterance_type not in ['linguistic', 'linguistic_lf0']:
            return utt_conds
        embedded_features = []
        for i in [2, 3, 4, 5, 6]:
            embedded_features.append(self.conds_utt_phonemes_emb(utt_conds[:, :, i].long()))
        embedded_features.append(self.conds_utt_vowels_emb(utt_conds[:, :, 27].long()))
        for i in [31, 33, 41]:
            embedded_features.append(self.conds_utt_gpos_emb(utt_conds[:, :, i].long()))
        embedded_features.append(self.conds_utt_tobi_emb(utt_conds[:, :, 49].long()))
        embedded_features.append(utt_conds[:, :, 0:2])
        embedded_features.append(utt_conds[:, :, 7:27])
        embedded_features.append(utt_conds[:, :, 28:31])
        embedded_features.append(utt_conds[:, :, 32:33])
        embedded_features.append(utt_conds[:, :, 34:41])
        embedded_features.append(utt_conds[:, :, 42:49])
        embedded_features.append(utt_conds[:, :, 50:])
        return torch.cat(embedded_features, dim=2)


class FrameLevelLayer(torch.nn.Module):
    input_samples = None
    ratio = None
    rnn_hidden_size = None

    def __init__(self, input_samples, conds_size, ratio, rnn_layers, rnn_hidden_size):
        super(FrameLevelLayer, self).__init__()
        self.input_samples = input_samples
        self.ratio = ratio
        self.rnn_hidden_size = rnn_hidden_size
        self.x_expand = torch.nn.Conv1d(input_samples, rnn_hidden_size, 1)
        self.conds_expand = torch.nn.Conv1d(conds_size, rnn_hidden_size, 1)
        self.rnn = torch.nn.GRU(rnn_hidden_size, rnn_hidden_size, rnn_layers, batch_first=True)
        self.rnn_h0 = torch.nn.Parameter(torch.zeros(rnn_layers, rnn_hidden_size))
        self.upsample = torch.nn.ConvTranspose1d(rnn_hidden_size, rnn_hidden_size, ratio, stride=ratio, bias=False)
        self.upsample_bias = torch.nn.Parameter(torch.zeros(rnn_hidden_size, ratio))
        self.upsample.reset_parameters()
        self._init_weight_norm()

    def _init_weight_norm(self):
        self.x_expand = torch.nn.utils.weight_norm(self.x_expand)
        self.conds_expand = torch.nn.utils.weight_norm(self.conds_expand)
        self.upsample = torch.nn.utils.weight_norm(self.upsample)

    def forward(self, x, conds, upper_conditioning, rnn_state):
        b, t, _ = x.size()
        if t != conds.shape[1]:
            upscale_ratio = int(x.shape[1] / conds.shape[1])
            conds = conds.unsqueeze(2).expand(b, conds.shape[1], upscale_ratio, conds.shape[2]) \
                .reshape(b, t, conds.shape[2])
        x = self.x_expand(x.permute(0, 2, 1)).permute(0, 2, 1)
        conds = self.conds_expand(conds.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + conds + upper_conditioning if upper_conditioning is not None else x + conds
        hidden_state_tensor = torch.cat([
            self.rnn_h0.unsqueeze(1) if state is None else state.unsqueeze(1) for _, state in enumerate(rnn_state)
        ], dim=1)
        rnn_output, rnn_state_new = self.rnn(x, hidden_state_tensor)
        upsampling_bias = self.upsample_bias.unsqueeze(0).unsqueeze(2).expand(b, self.rnn_hidden_size, t, self.ratio) \
            .contiguous().view(b, self.rnn_hidden_size, t * self.ratio)
        upsampling_output = (self.upsample(rnn_output.permute(0, 2, 1)) + upsampling_bias).permute(0, 2, 1)
        return upsampling_output, rnn_state_new


class SampleLevelLayer(torch.nn.Module):
    input_samples = None
    q_levels = None

    def __init__(self, input_samples, conds_size, rnn_hidden_size, q_levels):
        super(SampleLevelLayer, self).__init__()
        self.input_samples = input_samples
        self.q_levels = q_levels
        self.emb_layer = torch.nn.Embedding(q_levels, q_levels)
        self.emb_layer_expand = torch.nn.Conv1d(q_levels, rnn_hidden_size, input_samples, bias=False)
        self.conds_expand = torch.nn.Conv1d(conds_size, rnn_hidden_size, 1)
        self.comb_layer = torch.nn.Linear(rnn_hidden_size * 3, rnn_hidden_size)
        self.comb_layer_expand = torch.nn.Conv1d(rnn_hidden_size, rnn_hidden_size, 1)
        self.adapt = torch.nn.Conv1d(rnn_hidden_size, q_levels, 1)
        self._init_weight_norm()

    def _init_weight_norm(self):
        self.emb_layer_expand = torch.nn.utils.weight_norm(self.emb_layer_expand)
        self.comb_layer_expand = torch.nn.utils.weight_norm(self.comb_layer_expand)
        self.adapt = torch.nn.utils.weight_norm(self.adapt)

    def forward(self, x, conds, upper_tier_conditioning):
        upscale_ratio = int(upper_tier_conditioning.shape[1] / conds.shape[1])
        conds = conds.unsqueeze(2).expand(x.size(0), conds.shape[1], upscale_ratio, conds.shape[2]) \
            .reshape(x.size(0), upper_tier_conditioning.shape[1], conds.shape[2])
        embedding_output = self.emb_layer(x.contiguous().view(-1)).view(x.size(0), -1, self.q_levels)
        embedding_expand_output = self.emb_layer_expand(embedding_output.permute(0, 2, 1))
        conds_expand_output = self.conds_expand(conds.permute(0, 2, 1))
        inputs_comb_output = F.relu(
            self.comb_layer(torch.cat(
                (embedding_expand_output.permute(0, 2, 1), conds_expand_output.permute(0, 2, 1),
                 upper_tier_conditioning), dim=2)
            )
        )
        global_expand_output = F.relu(self.comb_layer_expand(inputs_comb_output.permute(0, 2, 1)))
        adaptation_output = self.adapt(global_expand_output)
        return F.log_softmax(adaptation_output.permute(0, 2, 1), dim=2)


class SampleRNNModel(torch.nn.Module):
    receptive_field = None
    quantizer = None
    rnn_states = None

    def __init__(self, conds_speaker_type, conds_speaker_n, conds_speaker_size, conds_utterance_type,
                 conds_utterance_linguistic_n, conds_utterance_linguistic_emb_size, conds_size, sequence_length, ratios,
                 rnn_layers, rnn_hidden_size, q_type_ulaw, q_levels):
        super(SampleRNNModel, self).__init__()
        self.receptive_field = np.prod(ratios) * sequence_length
        self.quantizer = utils.SampleRNNQuantizer(q_type_ulaw, q_levels)

        # Initialize conditionants mixer
        self.conds_mixer = CondsMixer(conds_speaker_type, conds_speaker_n, conds_speaker_size, conds_utterance_type,
                                      conds_utterance_linguistic_n, conds_utterance_linguistic_emb_size, conds_size)

        # Initialize frame level layers
        self.frames_layers = torch.nn.ModuleList()
        frame_layers_fs = list(map(int, np.cumprod(ratios)))
        for layer_n in range(0, len(frame_layers_fs)):
            self.frames_layers.append(
                FrameLevelLayer(frame_layers_fs[layer_n], conds_size, ratios[layer_n], rnn_layers[layer_n],
                                rnn_hidden_size[layer_n])
            )

        # Initialize sample level layer
        self.sample_layer = SampleLevelLayer(ratios[0], conds_size, rnn_hidden_size[0], self.quantizer.q_levels)

    def _init_rnn_states(self, batch_size):
        self.rnn_states = {rnn: [None] * batch_size for rnn in self.frames_layers}

    def _get_rnn_states(self, layer, reset):
        return [
            self.rnn_states[layer][reset_index] if reset_element == 0 else None
            for reset_index, reset_element in enumerate(reset)
        ]

    def _set_rnn_states(self, new_hidden_state_tensor, frame_level_layer, reset):
        last_hidden_state = 0
        for reset_index, reset_element in enumerate(reset):
            if reset_element == 0 or reset_element == 1:
                self.rnn_states[frame_level_layer][reset_index] = \
                    new_hidden_state_tensor[:, last_hidden_state, :]
                last_hidden_state += 1
            else:
                self.rnn_states[frame_level_layer][reset_index] = None

    def forward(self, x, y, utt_conds, info, reset):
        b, t, _ = utt_conds.size()

        # Init RNN states, if not done
        if not hasattr(self, 'rnnstates'):
            self._init_rnn_states(b)

        # Clean the utterances with reset == 2 (empty sample). Handle multi-gpu scenario, where one GPU may receive
        # all it samples with reset == 2.
        if all(reset == 2):
            return torch.zeros(x.size(0), self.receptive_field, self.quantizer.q_levels).to(x.device)
        else:
            x = x[reset != 2]
            y = y[reset != 2]
            utt_conds = utt_conds[reset != 2]

        # Quantize both x and y
        x, y = self.quantizer.quantize(x), self.quantizer.quantize(y)

        # Mix both the speaker and utterance conditionants
        conds = self.conds_mixer(utt_conds, info)

        # Propagate through frame level layers
        upper_tier_conditioning = None
        for layer in reversed(self.frames_layers):
            from_index = self.frames_layers[-1].input_samples - layer.input_samples
            to_index = -layer.input_samples + 1
            input_samples = self.quantizer.dequantize(x[:, from_index: to_index])
            input_samples = input_samples.contiguous().view(x.size(0), -1, layer.input_samples)
            rnn_states = self._get_rnn_states(layer, reset)
            upper_tier_conditioning, rnn_states_new = layer(
                input_samples, conds, upper_tier_conditioning, rnn_states
            )
            self._set_rnn_states(rnn_states_new.detach(), layer, reset)

        # Propagate through sample level layers
        input_samples = x[:, (self.frames_layers[-1].input_samples - self.sample_layer.input_samples):]
        y_hat = self.sample_layer(input_samples, conds, upper_tier_conditioning)

        # Return both y_hat and y, even this last one is not used (just quantized for loss computation)
        return y_hat, y

    def test(self, utt_conds, reset):
        b, t, conds_size = utt_conds.size()

        # Create a Tensor to store the generated samples in
        generated_sequences = torch.zeros(b, (t + 1) * self.conf.architecture['frame_size'], dtype=torch.int64
                                          ).fill_(self.quantizer.quantize_zero()).to(utt_conds.device)

        # Create a list to store the conditioning
        frame_level_outputs = [None for _ in self.frames_layers]

        # Iterate over the samples
        for xi in range(self.conf.architecture['frame_size'], generated_sequences.shape[1]):
            # Compute conds index
            conds_indx, _ = divmod(xi, self.conf.architecture['frame_size'])
            conds_indx -= 1

            # On
            if xi == self.conf.architecture['frame_size'] + 1:
                reset[reset == 1] = 0

            # Iterate over Frame Level layers
            for (frame_level_indx, frame_level_layer) in reversed(list(enumerate(self.frame_level_layers))):

                # If the generated sample is not a multiple of the input size, skip
                if xi % frame_level_layer.frame_input_samples != 0:
                    continue

                # Prepare the input samples to enter the model
                frame_layer_input_samples = torch.autograd.Variable(self.quantizer.dequantize(
                    generated_sequences[:, xi - frame_level_layer.frame_input_samples:xi]
                ).unsqueeze(1))

                # Mode the variable to CUDA, if available
                if utt_conds.is_cuda:
                    frame_layer_input_samples = frame_layer_input_samples.cuda()

                # Check if we have conditioning
                if frame_level_indx == len(self.frame_level_layers) - 1:
                    upper_tier_conditioning = None

                # If we are not in the last tier
                else:

                    # Compute frame_index
                    frame_index = (xi // frame_level_layer.frame_input_samples) % \
                                  self.frame_level_layers[frame_level_indx + 1].frame_ratio

                    # Get the upper tier conditioning from the previous upper tier
                    upper_tier_conditioning = frame_level_outputs[frame_level_indx + 1][:, frame_index, :] \
                        .unsqueeze(1)

                # Set the new hidden states
                frame_level_hidden_state = self._get_frame_level_hidden_states(
                    frame_level_layer=frame_level_layer,
                    reset_list=reset
                )

                # Propagate through current frame level layer
                frame_level_outputs[frame_level_indx], new_frame_level_hiddden_state = \
                    frame_level_layer(
                        input_samples=frame_layer_input_samples,
                        input_conds=utt_conds[:, conds_indx, :].unsqueeze(1),
                        upper_tier_conditioning=upper_tier_conditioning,
                        rnn_hidden_state=frame_level_hidden_state
                    )

                # Set the new frame level hidden state
                self._set_frame_level_hidden_states(
                    new_hidden_state_tensor=new_frame_level_hiddden_state.detach(),
                    frame_level_layer=frame_level_layer,
                    reset_list=reset
                )

            # Prepare the input samples Sample Level Layer
            sample_layer_input_samples = \
                generated_sequences[:, xi - self.sample_level_layer.frame_input_samples:xi]

            # Mode the variable to CUDA, if available
            if utt_conds.is_cuda:
                sample_layer_input_samples = sample_layer_input_samples.cuda()

            # Prepare conditioning
            upper_tier_conditioning = frame_level_outputs[0][:, xi % self.sample_level_layer
                .frame_input_samples, :].unsqueeze(1)

            # Store generated samples
            generated_sequences[:, xi] = self.sample_level_layer(
                input_samples=sample_layer_input_samples,
                input_conds=utt_conds[:, conds_indx, :].unsqueeze(1),
                upper_tier_conditioning=upper_tier_conditioning
            ).squeeze(1).exp_().multinomial(1).squeeze(1)

        # Return generated samples
        return generated_sequences
