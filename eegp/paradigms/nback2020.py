"""
Paradigm: Nback 2020 
Authors: Matthew Tan <5636374@qq.com>
Update time: 2021.4.20
"""
import os
import numpy as np
import pandas as pd

from mne import Epochs, events_from_annotations
from mne import pick_events
from mne import Epochs, events_from_annotations

from .base import BaseParadigm
from .utils import read_raw, remove_eog_template_ica, remove_eog_ica, channel_repair_exclud
from ..utils import _edge_index, _check_paths


class NBack(BaseParadigm):
    """This class provides preprocess pipeline and should never be instantiated
    directly.
    """
    def __init__(self,
                 code,
                 tmin=0,
                 tmax=1,
                 filter_low=0.5,
                 filter_high=None,
                 resample=250,
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
        super(NBack, self).__init__(code=code)
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
            raw = channel_repair_exclud(raw,
                                        exclude=[],
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

    def _remove_eog(self, raws):
        if len(raws) > 1:
            raws = remove_eog_template_ica(raws,
                                           n_components=10,
                                           ch_name='F3',
                                           threshold=0.8)
        else:
            raws = [
                remove_eog_ica(raws[0],
                               n_components=10,
                               ch_name='F3',
                               threshold=3)
            ]
        return raws

    def _define_trials(self, raw):
        memory_items = {'10', '11'}
        events_raw, event_id_raw = events_from_annotations(raw)
        event_id_new = {
            key: event_id_raw[key]
            for key in event_id_raw if key in memory_items
        }
        events_new = pick_events(events_raw,
                                 include=list(event_id_new.values()))
        return events_new, event_id_new

    def _metadata_from_raw(self, epochs, raw):
        nback_id = {
            '1': set([1, 2, 3, 4, 5, 6, 13, 14]),
            '2': set([7, 8, 9, 10, 11, 12, 15, 16])
        }
        type_id = {
            'number': set([1, 2, 7, 8]),
            'sound': set([3, 4, 9, 10, 13, 14, 15, 16]),
            'shape': set([5, 6, 11, 12, 13, 14, 15, 16])
        }

        file_edges = _edge_index(raw)
        file_2_run = lambda x: int(os.path.split(os.path.split(x)[0])[1])

        # help function
        def index_2_run(index):
            for filename, (onset, end) in file_edges.items():
                if onset <= index <= end:
                    run = file_2_run(filename)
                    return run

        def index_2_nback(index):
            run = index_2_run(index)
            for nback, runs in nback_id.items():
                if run in runs:
                    return nback

        def index_2_type(index):
            run = index_2_run(index)
            for item_type, runs in type_id.items():
                if run in runs:
                    return item_type

        columns = ['Sample index', 'Nback', 'Type']
        metadata = pd.DataFrame(columns=columns)
        for i in range(len(epochs.events)):
            index = epochs.events[i][0]
            nback = index_2_nback(index)
            item_type = index_2_type(index)
            metadata.loc[i] = np.array([index, nback, item_type])

        metadata['Sample index'] = metadata['Sample index'].map(int)
        metadata['Nback'] = metadata['Nback'].map(int)

        return metadata


class OneBack(NBack):
    def __init__(self, *args, **kwargs):
        super(OneBack, self).__init__(code='1-back', *args, **kwargs)

    def make_data(self):
        if not self.epochs:
            self.make_epochs()

        self._datas = []
        self._labels = []
        for epoch in self.epochs:
            self._datas.append(epoch['Nback == 1'].copy().crop(
                tmin=self.tmin, tmax=self.tmax,
                include_tmax=False).get_data().astype(np.float32))
            self._labels.append(
                epoch['Nback == 1'].metadata['Nback'].to_numpy())


class TwoBack(NBack):
    def __init__(self, *args, **kwargs):
        super(TwoBack, self).__init__(code='2-back', *args, **kwargs)

    def make_data(self):
        if not self.epochs:
            self.make_epochs()

        self._datas = []
        self._labels = []
        for epoch in self.epochs:
            self._datas.append(epoch['Nback == 2'].copy().crop(
                tmin=self.tmin, tmax=self.tmax,
                include_tmax=False).get_data().astype(np.float32))
            self._labels.append(
                epoch['Nback == 2'].metadata['Nback'].to_numpy())
