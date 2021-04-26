"""
Paradigm: PhysioNet MI
Website: https://physionet.org/content/eegmmidb/1.0.0/
Authors: Matthew Tan <5636374@qq.com>
Update time: 2021.4.20
"""
import os

import numpy as np
import pandas as pd

from mne import Epochs
from mne.datasets import eegbci

from .base import BaseParadigm
from ..path import check_paths
from ..core import read_raw
from ..core import remove_eog_template_ica, remove_eog_ica, channel_repair_exclud
from ..core import get_raw_edge_index


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
        super(PhysioNetMI, self).__init__(code=code)
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
            self._raws = self._remove_eog(self._raws)

    def make_epochs(self):
        if not self._raws:
            raise RuntimeError(
                'File haven\'t loaded yet, please load file first.')
        self._epochs = []
        for raw in self._raws:
            events_trials, event_id_trials = self._define_trials(
                raw, ['T1', 'T2'])  # T1 T2 represent the stimulus
            epochs = Epochs(raw,
                            events_trials,
                            event_id_trials,
                            self.tmin - 0.2,
                            self.tmax + 0.2,
                            baseline=self.baseline,
                            reject=dict(eeg=self.reject),
                            preload=True)
            epochs.metadata = self._metadata_from_raw(epochs, raw)
            epochs = self._filter_epochs(epochs)
            epochs.metadata = self._make_metadata(epochs.metadata)
            self._epochs.append(epochs.resample(self.resample))

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

    def _metadata_from_raw(self, epochs, raw):
        runs_id = {
            'hands vs feet': set([6, 10, 14]),
            'left vs right': set([4, 8, 12])
        }

        file_edges = get_raw_edge_index(raw)
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

        columns = ['Task']
        metadata = pd.DataFrame(columns=columns)
        for i in range(len(epochs.events)):
            index = epochs.events[i][0]
            task_type = index_2_type(index, epochs.events[i][2])
            metadata.loc[i] = np.array([task_type])

        return metadata

    def _make_metadata(self, metadata):
        label_map = {'hands': 0, 'feet': 1, 'left hands': 2, 'right hands': 3}
        metadata['Label'] = metadata['Task'].map(label_map)
        return metadata


class MIFeetHand(PhysioNetMI):
    """Imagine raising feet or making fists.
    """
    def __init__(self, *args, **kwargs):
        super(MIFeetHand, self).__init__(code='Feet&Hands-PhysioNet',
                                         *args,
                                         **kwargs)

    def make_data(self):
        if not self.epochs:
            self.make_epochs()
        self._datas = []
        self._labels = []
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
        super(MILeftRight, self).__init__(code='Left&Right-PhysioNet',
                                          *args,
                                          **kwargs)

    def make_data(self):
        if not self.epochs:
            self.make_epochs()

        self._datas = []
        self._labels = []
        for epoch in self.epochs:
            self._datas.append(epoch['Label == 3 or Label == 4'].copy().crop(
                tmin=self.tmin, tmax=self.tmax,
                include_tmax=False).get_data().astype(np.float32))
            self._labels.append(
                epoch['Label == 3 or Label == 4'].metadata['Label'].to_numpy())
