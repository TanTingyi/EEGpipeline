"""
Paradigm: PhysioNet MI
Website: https://physionet.org/content/eegmmidb/1.0.0/
Authors: Matthew Tan <5636374@qq.com>
Update time: 2021.4.20
"""
import os

import numpy as np

from mne import Epochs, events_from_annotations
from mne.datasets import eegbci
from mne.epochs import make_metadata

from .base import BaseParadigm
from .utils import transform_event_id
from ..path import check_paths
from ..core.load import read_raw
from ..core.preprocess import (remove_eog_template_ica, remove_eog_ica,
                               channel_repair_exclud, band_pass_filter)
from ..core.metadata import get_raw_edge_index


class PhysioNetMI(BaseParadigm):
    """This class provides preprocess pipeline and should never be instantiated
    directly.
    """
    def __init__(self,
                 tmin,
                 tmax,
                 filter_low=7.,
                 filter_high=30.,
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
        remove_eog : bool
            Whether to remove EOG artifacts.

        """
        super(PhysioNetMI, self).__init__(code='PhysioNet-MI')
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
            eegbci.standardize(raw)  # set channel names
            # strip channel names of "." characters
        self._raws = channel_repair_exclud(self._raws,
                                           exclude='bads',
                                           montage='standard_1005')
        self._raws = band_pass_filter(self._raws, self.filter_low,
                                      self.filter_high)
        if self.remove_eog:
            self._raws = self._remove_eog(self._raws)

    def make_epochs(self):
        if not self._raws:
            raise RuntimeError(
                'File haven\'t loaded yet, please load file first.')
        self._epochs = []
        for raw in self._raws:
            events, event_id, metadata = self._metadata_from_raw(raw)
            epochs = Epochs(raw,
                            events,
                            event_id,
                            self.tmin - 0.5,
                            self.tmax + 0.2,
                            baseline=self.baseline,
                            metadata=metadata,
                            preload=True)

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

    def _metadata_from_raw(self, raw):
        self._transform_event_id(raw)
        all_events, all_event_id = events_from_annotations(raw)

        row_events = [
            'stimulus/imagine/hands/all', 'stimulus/imagine/feet/all',
            'stimulus/imagine/hands/left', 'stimulus/imagine/hands/right'
        ]
        keep_first = ['stimulus']
        metadata_tmin, metadata_tmax = 0.0, 4

        metadata, events, event_id = make_metadata(events=all_events,
                                                   event_id=all_event_id,
                                                   tmin=metadata_tmin,
                                                   tmax=metadata_tmax,
                                                   sfreq=raw.info['sfreq'],
                                                   row_events=row_events,
                                                   keep_first=keep_first)
        # all times of the time-locked events should be zero
        assert all(metadata['stimulus'] == 0)
        return events, event_id, metadata

    def _transform_event_id(self, raw):
        def description_transform(raw):
            new_event_id = {
                'rest': 0,
                'stimulus/imagine/hands/all': 1,
                'stimulus/imagine/feet/all': 2,
                'stimulus/imagine/hands/left': 3,
                'stimulus/imagine/hands/right': 4
            }
            all_events, all_event_id = events_from_annotations(raw)
            file_edges = get_raw_edge_index(raw)

            # help function
            def index_2_run(index):
                for filename, (onset, end) in file_edges.items():
                    if onset <= index <= end:
                        run = int(os.path.split(filename)[1].split('R')[1][:2])
                        return run

            def description_transform_by_run(run, old_description):
                runs_id = {
                    'hands&feet': set([6, 10, 14]),
                    'left&right': set([4, 8, 12])
                }
                for run_type, runs in runs_id.items():
                    if run in runs:
                        if run_type == 'hands&feet':
                            if old_description == all_event_id['T1']:
                                return 1
                            elif old_description == all_event_id['T2']:
                                return 2
                            elif old_description == all_event_id['T0']:
                                return 0
                        elif run_type == 'left&right':
                            if old_description == all_event_id['T1']:
                                return 3
                            elif old_description == all_event_id['T2']:
                                return 4
                            elif old_description == all_event_id['T0']:
                                return 0

            def mapfunc(event):
                run = index_2_run(event[0])
                new_description = description_transform_by_run(run, event[2])
                return [event[0], event[1], new_description]

            new_events = np.array(list(map(mapfunc, all_events)))
            return new_events, new_event_id

        transform_event_id(raw, description_transform=description_transform)

    def make_data(self):
        if not self.epochs:
            self.make_epochs()

        self._datas = []
        for epoch in self.epochs:
            self._datas.append(epoch.copy().crop(
                tmin=self.tmin, tmax=self.tmax,
                include_tmax=False).get_data().astype(np.float32))
