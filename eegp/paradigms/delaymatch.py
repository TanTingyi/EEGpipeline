"""
Paradigm: Letter delay match 2017
Paper: A Fusion Feature for Enhancing the Performance 
    of Classification in Working Memory Load 
    With Single-Trial Detection
DOI: 10.1109/TNSRE.2019.2936997
Authors: Matthew Tan <5636374@qq.com>
Update time: 2021.4.20
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import quantile_transform
from mne import Epochs, events_from_annotations, merge_events
from mne.event import define_target_events

from .base import BaseParadigm
from ..path import check_paths
from ..core import read_raw
from ..core import remove_eog_template_ica, remove_eog_ica, channel_repair_exclud


class LetterDelayMatch(BaseParadigm):
    """This class provides preprocess pipeline and should never be instantiated
    directly.
    """
    def __init__(self,
                 code,
                 tmin,
                 tmax,
                 filter_low=0.5,
                 filter_high=None,
                 resample=160,
                 baseline=None,
                 remove_eog=False):
        """
        Parameters
        ----------
        tmin : int | float
            Start time before event. If nothing is provided, 
            defaults to -0.2.
        tmax : int | float
            End time after event. If nothing is provided, 
            defaults to 0.5.
        filter_low : None | int | float
            For FIR filters, the lower pass-band edge; 
            If None the data are only low-passed.
        filter_high : None | int | float
            For FIR filters, the upper pass-band edge; 
            If None the data are only high-passed.
        resample : int
            New sample rate to use.
        baseline : None | tuple
            The time interval to consider as “baseline” when 
            applying baseline correction. If None, do not apply 
            baseline correction. 
        reject : None | int
            Reject epochs based on peak-to-peak signal amplitude.
            unit: V
        remove_eog : bool
            Whether to remove EOG artifacts.

        """
        super(LetterDelayMatch, self).__init__(code=code)
        self.tmin = tmin
        self.tmax = tmax
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.resample = resample
        self.baseline = baseline
        self.remove_eog = remove_eog

    def read_raw(self, paths):
        self._paths = check_paths(paths)
        self._raws = read_raw(self._paths)

    def preprocess(self):
        if not self._raws:
            raise RuntimeError('File not Loaded.')

        for raw in self._raws:
            raw = channel_repair_exclud(
                raw,
                exclude=['CB1', 'CB2', 'HEO', 'VEO', 'M1', 'M2'],
                montage='standard_1020')
            raw.filter(self.filter_low,
                       self.filter_high,
                       skip_by_annotation='edge')

        if self.remove_eog:
            self._raws = self._remove_eog(self._raws)

    def make_epochs(self):
        if not self._raws:
            raise RuntimeError(
                'File haven\'t loaded yet, please load file first.')
        self._epochs = []
        for raw in self._raws:
            events_trials, event_id_trials = self._define_trials(
                raw, ['22', '44', '88'])  # 22 44 88 represent the stimulus
            epochs = Epochs(raw,
                            events_trials,
                            event_id_trials,
                            self.tmin - 0.2,
                            self.tmax + 0.2,
                            baseline=self.baseline,
                            preload=True)
            epochs.metadata = self._metadata_from_raw(epochs, raw)
            epochs = self._filter_epochs(epochs)
            epochs.metadata = self._make_metadata(epochs.metadata)
            self._epochs.append(epochs.resample(self.resample))

    def _remove_eog(self, raws):
        if len(raws) > 1:
            raws = remove_eog_template_ica(raws,
                                           n_components=15,
                                           ch_name='FPZ',
                                           threshold=0.8)
        else:
            raws = [
                remove_eog_ica(raws[0],
                               n_components=15,
                               ch_name='FPZ',
                               threshold=3)
            ]
        return raws

    def _metadata_from_raw(self, epochs, raw):
        """...22---------------122-----------------1 or 3...
           ...|   maintenance   |   reaction time     |...
        """
        events, event_id = events_from_annotations(raw)
        sfreq = raw.info['sfreq']
        item_id = 30
        cue_id = 31
        correct_id_new = 32
        wrong_id_new = 33
        correct_id = event_id['1']
        wrong_id = event_id['3']

        # 1. merge event id.
        events_tmp = merge_events(
            events, [event_id['22'], event_id['44'], event_id['88']],
            new_id=item_id)
        events_tmp = merge_events(
            events_tmp, [event_id['122'], event_id['144'], event_id['188']],
            new_id=cue_id)

        # 2. clacluate maintenance between memory item and cue
        events_maintenance, maintenance_lag = define_target_events(events_tmp,
                                                                   item_id,
                                                                   cue_id,
                                                                   sfreq,
                                                                   tmin=3.2,
                                                                   tmax=4.0)
        maintenance_array = np.stack(
            [events_maintenance[:, 0], maintenance_lag / 1000], axis=1)

        # 3. clacluate duration between item and button
        events_trial_correct, trial_lag_correct = define_target_events(
            events_tmp,
            item_id,
            correct_id,
            sfreq,
            tmin=3.2,
            tmax=5.5,
            new_id=correct_id_new)
        events_trial_wrong, trial_lag_wrong = define_target_events(
            events_tmp,
            item_id,
            wrong_id,
            sfreq,
            tmin=3.2,
            tmax=5.5,
            new_id=wrong_id_new)

        events_trial = np.concatenate(
            [events_trial_correct, events_trial_wrong], axis=0)
        trial_lag = np.concatenate([trial_lag_correct, trial_lag_wrong],
                                   axis=0)
        trial_array = np.stack([events_trial[:, 0], trial_lag / 1000], axis=1)

        # help function
        def id_2_item(id):
            if id == event_id['22']:
                return 2
            elif id == event_id['44']:
                return 4
            elif id == event_id['88']:
                return 8

        def find_by_index(index, array):
            return array[array[:, 0] == index]

        # 4. create metadata from array
        columns = ['Maintenance', 'Correct', 'Trial time', 'Item amount']
        metadata = pd.DataFrame(columns=columns)
        for i in range(len(epochs.events)):
            maintenance = find_by_index(epochs.events[i][0],
                                        maintenance_array)[0][1]
            trial_time = find_by_index(epochs.events[i][0], trial_array)[0][1]
            correct = find_by_index(epochs.events[i][0], events_trial)[0][2]
            item_amount = id_2_item(epochs.events[i][2])
            metadata_array = np.array(
                [maintenance, correct, trial_time, item_amount])
            metadata.loc[i] = metadata_array

        metadata[
            'Reaction time'] = metadata['Trial time'] - metadata['Maintenance']

        # 6. type transform
        metadata["Correct"] = metadata["Correct"].map({
            correct_id_new: True,
            wrong_id_new: False
        })
        metadata["Item amount"] = metadata["Item amount"].map(int)

        return metadata

    def _make_metadata(self, metadata):
        # add srt
        reaction_time_max = 1.617
        srt = lambda x: reaction_time_max - x['Reaction time'] if x[
            'Correct'] else x['Reaction time'] - reaction_time_max
        metadata['SRT'] = metadata.apply(srt, axis=1)
        metadata['SRT_normal'] = quantile_transform(
            metadata['SRT'].to_numpy().reshape(-1, 1),
            output_distribution='normal',
            copy=True).squeeze()
        metadata['SRT_bins_freq'] = pd.qcut(metadata['SRT'],
                                            3,
                                            labels=[0, 0.5, 1])
        metadata['SRT_bins_width'] = pd.cut(metadata['SRT'],
                                            3,
                                            labels=[0, 0.5, 1])
        metadata['SRT_normal_bins_width'] = pd.cut(metadata['SRT_normal'],
                                                   3,
                                                   labels=[0, 0.5, 1])
        return metadata

    def _filter_epochs(self, epochs):
        return epochs['Maintenance > 3.3']


class Letter248(LetterDelayMatch):
    """延时匹配-字母，根据刺激的数量定义标签
    """
    def __init__(self, *args, **kwargs):
        super(Letter248, self).__init__(code='Delay Match Letter 248',
                                        *args,
                                        **kwargs)

    def make_data(self):
        if not self.epochs:
            self.make_epochs()

        self._datas = []
        self._labels = []
        for epoch in self.epochs:
            self._datas.append(epoch.copy().crop(
                tmin=self.tmin, tmax=self.tmax,
                include_tmax=False).get_data().astype(np.float32))
            self._labels.append(epoch.metadata['Item amount'].to_numpy())


class LetterSRT(LetterDelayMatch):
    """延时匹配-字母，根据SRT定义标签
    """
    def __init__(self, *args, **kwargs):
        super(LetterSRT, self).__init__(code='Delay Match Letter SRT',
                                        *args,
                                        **kwargs)

    def _filter_epochs(self, epochs):
        return epochs['Maintenance > 3.5']

    def make_data(self):
        if not self.epochs:
            self.make_epochs()

        self._datas = []
        self._labels = []
        for epoch in self.epochs:
            self._datas.append(epoch.copy().crop(
                tmin=self.tmin, tmax=self.tmax,
                include_tmax=False).get_data().astype(np.float32))
            self._labels.append(epoch.metadata['SRT_bins_freq'].to_numpy())
