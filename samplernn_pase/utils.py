import re
import ahoproc_tools
import numpy as np
from concurrent.futures import ThreadPoolExecutor

lab_regex = r'([0-9]+) ([0-9]+) '
lab_regex += r'(.+)\^(.+)-(.+)\+(.+)=(.+)@(.+)_(.+)'
lab_regex += r'/A:(.+)_(.+)_(.+)'
lab_regex += r'/B:(.+)-(.+)-(.+)@(.+)-(.+)&(.+)-(.+)#(.+)-(.+)\$(.+)-(.+)!(.+)-(.+);(.+)-(.+)\|(.+)'
lab_regex += r'/C:(.+)\+(.+)\+(.+)'
lab_regex += r'/D:(.+)_(.+)'
lab_regex += r'/E:(.+)\+(.+)@(.+)\+(.+)&(.+)\+(.+)#(.+)\+(.+)'
lab_regex += r'/F:(.+)_(.+)'
lab_regex += r'/G:(.+)_(.+)'
lab_regex += r'/H:(.+)\=(.+)@(.+)=(.+)\|(.+)'
lab_regex += r'/I:(.+)=(.+)'
lab_regex += r'/J:(.+)\+(.+)-(.+)'


def read_lab(lab_file_path):
    lab_read_lines = []
    with open(lab_file_path) as lab_file:
        for lab_line in lab_file.readlines():
            lab_read_lines.append(re.search(lab_regex, lab_line).groups())
    return lab_read_lines


class SampleRNNQuantizer:
    LINEAR_QUANT = 0
    ULAW_QUANT = 1
    _EPSILON = 1e-2
    _EPSILONs = 1e-6
    _MU = 255.
    _LOG_MU1 = 5.5451774444795623
    q_type = None
    q_levels = None

    def __init__(self, q_type_ulaw, q_levels):
        self.q_type = self.ULAW_QUANT if q_type_ulaw else self.LINEAR_QUANT
        self.q_levels = q_levels

    def quantize_zero(self):
        return self.q_levels // 2

    def quantize(self, samples):
        if self.q_type == self.LINEAR_QUANT:
            return self.linear_quantize(samples=samples)
        elif self.q_type == self.ULAW_QUANT:
            return self.ulaw_quantize(samples=samples)
        else:
            raise NotImplemented('Unrecognized q_type')

    def dequantize(self, samples):
        if self.q_type == self.LINEAR_QUANT:
            return self.linear_dequantize(samples=samples)
        elif self.q_type == self.ULAW_QUANT:
            return self.ulaw_dequantize(samples=samples)
        else:
            raise NotImplemented('Unrecognized q_type')

    def linear_quantize(self, samples):
        samples = samples.clone()
        samples -= samples.min(dim=-1)[0].expand_as(samples)
        samples /= samples.max(dim=-1)[0].expand_as(samples)
        samples *= self.q_levels - self._EPSILON
        samples += self._EPSILON / 2
        return samples.long()

    def linear_dequantize(self, samples):
        return samples.float() / (self.q_levels / 2) - 1

    def ulaw_quantize(self, samples):
        return self.midrise(self.ulaw(samples))

    def ulaw_dequantize(self, samples):
        return self.iulaw(self.imidrise(samples))

    def ulaw(self, x, max_value=1.0):
        v = self._MU / max_value
        y = x.sign() * (v * x.abs() + 1.).log() / self._LOG_MU1
        return y

    def iulaw(self, c):
        x = (c.abs() * self._LOG_MU1).exp() - 1
        y = c.sign() * x / self._MU
        return y

    def midrise(self, x):
        x = 0.5 * (x + 1.0)
        x *= (self.q_levels - self._EPSILONs)
        return x.long()

    def imidrise(self, xq):
        return xq.float() * 2.0 / self.q_levels - 1.0


class SampleRNNAhocoder:

    @staticmethod
    def read_acoustic_conds(utterance_path: str):
        # Load the Conds Files
        generated_utterance_cc = ahoproc_tools.io.read_aco_file(utterance_path + '.cc', (-1, 40))
        generated_utterance_fv = ahoproc_tools.io.read_aco_file(utterance_path + '.fv', (-1,))
        generated_utterance_lf0 = ahoproc_tools.io.read_aco_file(utterance_path + '.lf0', (-1,))

        # Interpolate FV and LF0, obtain VU
        generated_utterance_fv, _ = ahoproc_tools.interpolate.interpolation(generated_utterance_fv, 1e3)
        generated_utterance_lf0, generated_utterance_vu = ahoproc_tools.interpolate.interpolation(
            generated_utterance_lf0, -1e10)

        # Unsqueeze dimensions
        generated_utterance_fv = np.expand_dims(generated_utterance_fv, axis=1)
        generated_utterance_lf0 = np.expand_dims(generated_utterance_lf0, axis=1)
        generated_utterance_vu = np.expand_dims(generated_utterance_vu, axis=1)

        # Return 43 conditionants
        return np.concatenate(
            (generated_utterance_cc, generated_utterance_fv, generated_utterance_lf0, generated_utterance_vu),
            axis=1
        )

    @staticmethod
    def generate_acoustic_conds(utterances_paths):
        with ThreadPoolExecutor(max_workers=50) as executor:
            for utterance_path in utterances_paths:
                executor.submit(ahoproc_tools.io.wav2aco, utterance_path)

    @staticmethod
    def do_compute_objective_metrics(original_parameters, generated_parameters):
        cc_mcd = ahoproc_tools.error_metrics.MCD(original_parameters[:, :40], generated_parameters[:, :40])
        lf0_rmse = ahoproc_tools.error_metrics.RMSE(np.exp(original_parameters[:, 41]),
                                                    np.exp(generated_parameters[:, 41]))
        vu_afpr = ahoproc_tools.error_metrics.AFPR(original_parameters[:, 42], generated_parameters[:, 42])
        return cc_mcd, lf0_rmse, vu_afpr

    @staticmethod
    def compute_objective_metrics(data_info, experiment, epoch_n, pase_seed_duration=None):
        # Create lists to store error metrics
        lf0_rmse = []
        vu_afpr = []
        cc_mcd = []

        # Iterate over all the utterances
        for data_info_item in data_info:
            # Get Paths of original and generated conds
            utterance_conds_path = data_info_item['dataset']['conds_utterance']['acoustic_folder_path'] + \
                                   data_info_item['utterance']['path']
            utterance_generated_conds_path = \
                experiment.generated_samples_path + experiment.get_generated_sample_name(data_info_item, epoch_n,
                                                                                         pase_seed_duration)

            # Get Acoustic conds from both files
            utterance_acoustic_conds = SampleRNNAhocoder.read_acoustic_conds(utterance_conds_path)
            utterance_generated_acoustic_conds = SampleRNNAhocoder.read_acoustic_conds(utterance_generated_conds_path)

            # Compute Measures
            cc_mcd.append(ahoproc_tools.error_metrics.MCD(
                utterance_acoustic_conds[:, :40], utterance_generated_acoustic_conds[:, :40]))
            lf0_rmse.append(ahoproc_tools.error_metrics.RMSE(
                np.exp(utterance_acoustic_conds[:, 41]), np.exp(utterance_generated_acoustic_conds[:, 41])))
            vu_afpr.append(ahoproc_tools.error_metrics.AFPR(
                utterance_acoustic_conds[:, 42], utterance_generated_acoustic_conds[:, 42]))

        # Return MEAN error metrics
        return np.mean(cc_mcd), np.mean(lf0_rmse), np.mean(vu_afpr, axis=0)
