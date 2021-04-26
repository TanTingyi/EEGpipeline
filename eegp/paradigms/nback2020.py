"""
Paradigm: Nback 2020 
Authors: Matthew Tan <5636374@qq.com>
Update time: 2021.4.20
"""
import os
import numpy as np
import pandas as pd

from mne import Epochs

from .base import BaseParadigm
from ..path import check_paths
from ..core import read_raw
from ..core import remove_eog_template_ica, remove_eog_ica, channel_repair_exclud
from ..core import get_raw_edge_index


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
        super(NBack, self).__init__(code=code)
        self.tmin = tmin
        self.tmax = tmax
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.resample = resample
        self.baseline = baseline
        self.reject = reject
        self.remove_eog = remove_eog

    def read_raw(self, paths):
        self._paths = check_paths(paths)
        self._raws = read_raw(self._paths)

    def preprocess(self):
        if not self._raws:
            raise RuntimeError('File not Loaded.')

        for raw in self._raws:
            raw = channel_repair_exclud(raw,
                                        exclude=[],
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
                raw, ['10', '11'])  # 10 and 11 represent the stimulus
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
            self._epochs.append(epochs.resample(self.resample))

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
        # index of different file
        file_edges = get_raw_edge_index(raw)
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

        columns = ['Nback', 'Type']
        metadata = pd.DataFrame(columns=columns)
        for i in range(len(epochs.events)):
            index = epochs.events[i][0]
            nback = index_2_nback(index)
            item_type = index_2_type(index)
            metadata.loc[i] = np.array([nback, item_type])

        metadata['Nback'] = metadata['Nback'].map(int)

        return metadata


class OneBack(NBack):
    def __init__(self, *args, **kwargs):
        super(OneBack, self).__init__(code='1-back-2020', *args, **kwargs)

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
        super(TwoBack, self).__init__(code='1-back-2020', *args, **kwargs)

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
