import samplernn_pase.model
import skeltorch
import torch


class SampleRNNPASERunner(skeltorch.Runner):
    quantizer = None
    speaker_embedding = None

    def init_model(self, device):
        self.model = samplernn_pase.model.SampleRNNModel(
            conds_speaker_type=self.experiment.configuration.get('conditionals', 'conds_speaker_type'),
            conds_speaker_n=len(self.experiment.data.speakers_info),
            conds_speaker_size=self.experiment.configuration.get('conditionals', 'conds_speaker_size'),
            conds_utterance_type=self.experiment.configuration.get('conditionals', 'conds_utterance_type'),
            conds_utterance_linguistic_n=self.experiment.data.get_conds_linguistic_size(),
            conds_utterance_linguistic_emb_size=
            self.experiment.configuration.get('conditionals', 'conds_utterance_linguistic_emb_size'),
            conds_size=self.experiment.configuration.get('conditionals', 'conds_size'),
            sequence_length=self.experiment.configuration.get('model', 'sequence_length'),
            ratios=self.experiment.configuration.get('model', 'ratios'),
            rnn_layers=self.experiment.configuration.get('model', 'rnn_layers'),
            rnn_hidden_size=self.experiment.configuration.get('model', 'rnn_hidden_size'),
            q_type_ulaw=self.experiment.configuration.get('model', 'q_type_ulaw'),
            q_levels=self.experiment.configuration.get('model', 'q_levels')
        ).to(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.experiment.configuration.get('training', 'lr')
        )

    def train_step(self, it_data, device):
        x, y, utt_conds, reset, info = it_data
        x, y, utt_conds = x.to(device), y.to(device), utt_conds.to(device)
        y_hat, y = self.model(x, y, utt_conds, info, reset)
        return torch.nn.functional.nll_loss(y_hat.view(-1, y_hat.size(2)), y.view(-1))

    def test(self, epoch, device):
        pass