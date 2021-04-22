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
from mne.event import define_target_events, pick_events

from .base import BaseParadigm
from .utils import read_raw, remove_eog_template_ica, remove_eog_ica, channel_repair_exclud
from ..utils import _check_paths


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
                 reject=None,
                 remove_eog=False):
        """
        Parameters
        ----------
        tmin : int | float
            提取数据的开始时间点
        tmax : int | float
            提取数据的结束时间点
        filter_low : None | int | float
            带通滤波器的下截止频率
        filter_high : None | int | float
            带通滤波器的上截止频率
        resample : int
            重采样的采样频率
        baseline : None | tuple
            基线矫正的时间段，如果是 None 则不做基线矫正
        reject : None | dict
            阈值去除的阈值，例如 dict(eeg=200e-6) 将在
            分 trail 的时候丢弃峰峰值为 200 mv 的数据段
            为 None 时不启用
        remove_eog : bool
            是否去除眼电伪迹，默认为 False

        """
        super(LetterDelayMatch, self).__init__(code=code)
        self.tmin = tmin
        self.tmax = tmax
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.resample = resample
        self.baseline = baseline
        self.reject = reject
        self.remove_eog = remove_eog
        self.__raws = []
        self.__epochs = []
        self.__paths = []
        self._datas = []
        self._labels = []

    @property
    def raws(self):
        return self.__raws

    @property
    def epochs(self):
        return self.__epochs

    @property
    def paths(self):
        return self.__paths

    def read_raw(self, paths):
        self.__paths = _check_paths(paths).copy()
        self.__raws = read_raw(self.__paths)

    def preprocess(self):
        if not self.__raws:
            raise RuntimeError('File not Loaded.')

        for raw in self.__raws:
            raw = channel_repair_exclud(
                raw,
                exclude=['CB1', 'CB2', 'HEO', 'VEO', 'M1', 'M2'],
                montage='standard_1020')
            raw.filter(self.filter_low,
                       self.filter_high,
                       skip_by_annotation='edge')

        if self.remove_eog:
            self.__raws = self._remove_eog(self.__raws)

    def make_epochs(self):
        if not self.__raws:
            raise RuntimeError(
                'File haven\'t loaded yet, please load file first.')
        self.__epochs = []
        for raw in self.__raws:
            events_trials, event_id_trials = self._define_trials(raw)
            epochs = Epochs(raw,
                            events_trials,
                            event_id_trials,
                            self.tmin - 0.2,
                            self.tmax + 0.2,
                            baseline=self.baseline,
                            reject=self.reject,
                            preload=True)
            epochs.metadata = self._metadata_from_raw(epochs, raw)
            epochs = self._filter_epochs(epochs)
            epochs.metadata = self._make_metadata(epochs.metadata)
            self.__epochs.append(epochs.resample(self.resample))
            del raw

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

    def _define_trials(self, raw):
        memory_items = {'22', '44', '88'}
        events_raw, event_id_raw = events_from_annotations(raw)
        event_id_new = {
            key: event_id_raw[key]
            for key in event_id_raw if key in memory_items
        }
        events_new = pick_events(events_raw,
                                 include=list(event_id_new.values()))
        return events_new, event_id_new

    def _metadata_from_raw(self, epochs, raw):
        """...22---------------122-----------------1/3...
           ...|   maintenance   |   reaction time   |...
        """
        events, event_id = events_from_annotations(raw)
        sfreq = raw.info['sfreq']
        reaction_correct = event_id['1']  # correct button press
        reaction_wrong = event_id['3']  # wrong button press
        item_id = 30
        cue_id = 31
        maintenance_id = 50
        correct_id = 41
        wrong_id = 42
        fill_na = 90

        # 1. merge event id.
        events_tmp = merge_events(
            events, [event_id['22'], event_id['44'], event_id['88']],
            new_id=item_id)
        events_tmp = merge_events(
            events_tmp, [event_id['122'], event_id['144'], event_id['188']],
            new_id=cue_id)

        # 2. clacluate maintenance between memory item and cue item
        events_maintenance, maintenance = define_target_events(
            events_tmp,
            item_id,
            cue_id,
            sfreq,
            tmin=3.2,
            tmax=4.0,
            new_id=maintenance_id,
            fill_na=fill_na)
        events_maintenance[:, 1] = maintenance

        # 3. clacluate duration of entire trial
        events_correct, trial_time_correct = define_target_events(
            events_tmp,
            item_id,
            reaction_correct,
            sfreq,
            tmin=3.2,
            tmax=5.5,
            new_id=correct_id,
            fill_na=fill_na)
        events_wrong, trial_time_wrong = define_target_events(events_tmp,
                                                              item_id,
                                                              reaction_wrong,
                                                              sfreq,
                                                              tmin=3.2,
                                                              tmax=5.5,
                                                              new_id=wrong_id,
                                                              fill_na=fill_na)
        events_tmp = np.concatenate([events_correct, events_wrong], axis=0)
        trial_time = np.concatenate([trial_time_correct, trial_time_wrong],
                                    axis=0)
        events_with_metadata = np.append(events_tmp,
                                         trial_time.reshape(-1, 1),
                                         axis=1)
        events_with_metadata = events_with_metadata[
            events_with_metadata[:, 2] != float(fill_na)]

        # 4. add maintenance to array
        for event_with_metadata in events_with_metadata[:]:
            event_with_metadata[1] = events_maintenance[
                events_maintenance[:, 0] == event_with_metadata[0]][0][1]

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

        # 5. create metadata from array
        columns = [
            'Sample index', 'Maintenance', 'Correct', 'Trial time',
            'Item amount'
        ]
        metadata = pd.DataFrame(columns=columns)
        for i in range(len(epochs.events)):
            metadata_array = find_by_index(epochs.events[i][0],
                                           events_with_metadata)
            item_amount = id_2_item(epochs.events[i][2])
            metadata_array = np.append(metadata_array, item_amount)
            metadata_array[1] = metadata_array[1] / sfreq
            metadata_array[3] = metadata_array[3] / sfreq

            metadata.loc[i] = metadata_array

        metadata[
            'Reaction time'] = metadata['Trial time'] - metadata['Maintenance']

        # 6. type transform
        metadata["Correct"] = metadata["Correct"].map({
            correct_id: True,
            wrong_id: False
        })
        metadata["Item amount"] = metadata["Item amount"].map(int)
        metadata["Sample index"] = metadata["Sample index"].map(int)

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
