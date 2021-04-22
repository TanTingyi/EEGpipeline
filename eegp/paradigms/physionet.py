"""
Paradigm: PhysioNet MI
Website: https://physionet.org/content/eegmmidb/1.0.0/
Authors: Matthew Tan <5636374@qq.com>
Update time: 2021.4.20
"""
import os

import numpy as np
import pandas as pd

from mne import Epochs, events_from_annotations
from mne.event import pick_events
from mne.datasets import eegbci

from .base import BaseParadigm
from .utils import read_raw, remove_eog_template_ica, remove_eog_ica, channel_repair_exclud
from ..utils import _edge_index


class PhysioNetMI(BaseParadigm):
    """This class provides preprocess pipeline and should never be instantiated
    directly.
    """
    def __init__(self,
                 code,
                 tmin,
                 tmax,
                 filter_low=7.,
                 filter_high=30.,
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
        """
        super(PhysioNetMI, self).__init__(code=code)
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
        self.__paths = paths.copy()
        self.__raws = read_raw(self.__paths)

    def preprocess(self):
        if not self.__raws:
            raise RuntimeError('File not Loaded.')

        for raw in self.__raws:
            eegbci.standardize(raw)  # set channel names
            # strip channel names of "." characters
            raw.rename_channels(lambda x: x.strip('.'))
            raw = channel_repair_exclud(raw,
                                        exclude='bads',
                                        montage='standard_1005')
            raw.filter(self.filter_low,
                       self.filter_high,
                       skip_by_annotation='edge')

        if self.remove_eog:
            self.__raws = self._remove_eog(self.__raws)

    def make_epochs(self):
        if not self.__raws:
            raise RuntimeError(
                'File haven\'t loaded yet, please load file first.')

        for raw in self.__raws:
            events, event_id = self._define_trials(raw)
            epochs = Epochs(raw,
                            events,
                            event_id,
                            self.tmin - 0.2,
                            self.tmax + 0.2,
                            baseline=self.baseline,
                            reject=self.reject,
                            preload=True)
            epochs.metadata = self._metadata_from_raw(epochs, raw)
            epochs = self._filter_epochs(epochs)
            epochs.metadata = self._make_metadata(epochs.metadata)
            self.__epochs.append(epochs.resample(self.resample))

    def _remove_eog(self, raws):
        if len(raws) > 1:
            raws = remove_eog_template_ica(raws,
                                           n_components=15,
                                           ch_name='Fpz',
                                           threshold=0.8)
        else:
            raws = [
                remove_eog_ica(raws[0],
                               n_components=15,
                               ch_name='Fpz',
                               threshold=3)
            ]
        return raws

    def _define_trials(self, raw):
        events_raw, event_id_raw = events_from_annotations(raw)
        event_id_new = {
            key: event_id_raw[key]
            for key in event_id_raw if key in {'T1', 'T2'}
        }
        events_new = pick_events(events_raw,
                                 include=list(event_id_new.values()))
        return events_new, event_id_new

    def _metadata_from_raw(self, epochs, raw):
        runs_id = {
            'hands vs feet': set([6, 10, 14]),
            'left vs right': set([4, 8, 12])
        }

        file_edges = _edge_index(raw)
        file_2_run = lambda x: int(os.path.split(x)[1].split('R')[1][:2])

        # help function
        def index_2_run(index):
            for filename, (onset, end) in file_edges.items():
                if onset <= index <= end:
                    run = file_2_run(filename)
                    return run

        def index_2_type(index, description):
            run = index_2_run(index)
            for run_type, runs in runs_id.items():
                if run in runs:
                    if run_type == 'hands vs feet':
                        return 'hands' if description == epochs.event_id[
                            'T1'] else 'feet'
                    elif run_type == 'left vs right':
                        return 'left hands' if description == epochs.event_id[
                            'T1'] else 'right hands'

        columns = ['Sample index', 'Task']
        metadata = pd.DataFrame(columns=columns)
        for i in range(len(epochs.events)):
            index = epochs.events[i][0]
            task_type = index_2_type(index, epochs.events[i][2])
            metadata.loc[i] = np.array([index, task_type])

        metadata['Sample index'] = metadata['Sample index'].map(int)

        return metadata

    def _make_metadata(self, metadata):
        label_map = {'hands': 0, 'feet': 1, 'left hands': 2, 'right hands': 3}
        metadata['Label'] = metadata['Task'].map(label_map)
        return metadata


class MIFeetHand(PhysioNetMI):
    """Imagine raising feet or making fists.
    """
    def __init__(self, *args, **kwargs):
        super(MIFeetHand, self).__init__(code='MI Feet And Hands',
                                         *args,
                                         **kwargs)

    def make_data(self):
        if not self.epochs:
            self.make_epochs()

        for epoch in self.epochs:
            self._datas.append(epoch['Label == 1 or Label == 2'].copy().crop(
                tmin=self.tmin, tmax=self.tmax,
                include_tmax=False).get_data().astype(np.float32))
            self._labels.append(
                epoch['Label == 1 or Label == 2'].metadata['Label'].to_numpy())


class MILeftRight(PhysioNetMI):
    """Imagine holding left or right hand.
    """
    def __init__(self, *args, **kwargs):
        super(MILeftRight, self).__init__(code='MI Left And Right',
                                          *args,
                                          **kwargs)

    def make_data(self):
        if not self.epochs:
            self.make_epochs()

        for epoch in self.epochs:
            self._datas.append(epoch['Label == 3 or Label == 4'].copy().crop(
                tmin=self.tmin, tmax=self.tmax,
                include_tmax=False).get_data().astype(np.float32))
            self._labels.append(
                epoch['Label == 3 or Label == 4'].metadata['Label'].to_numpy())
