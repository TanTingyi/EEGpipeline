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

from mne import Epochs, events_from_annotations
from mne.epochs import make_metadata

from .base import BaseParadigm
from .utils import transform_event_id
from ..path import check_paths
from ..core.load import read_raw
from ..core.preprocess import (remove_eog_template_ica, remove_eog_ica,
                               channel_repair_exclud, rest_reference,
                               band_pass_filter)


class LetterDelayMatch(BaseParadigm):
    """This class provides preprocess pipeline.
    """
    def __init__(self,
                 tmin,
                 tmax,
                 filter_low=1,
                 filter_high=None,
                 resample=250,
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
        super(LetterDelayMatch, self).__init__(code='Delay Match Letter')
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
        self._raws = channel_repair_exclud(
            self._raws,
            exclude=['CB1', 'CB2', 'HEO', 'VEO', 'M1', 'M2'],
            montage='standard_1020')
        self._raws = band_pass_filter(self._raws, self.filter_low,
                                      self.filter_high)
        if self.remove_eog:
            self._raws = self._remove_eog(self._raws)
        self._raws = rest_reference(self._raws)

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

    def make_data(self):
        if not self.epochs:
            self.make_epochs()

        self._datas = []
        for epoch in self.epochs:
            self._datas.append(epoch.copy().crop(
                tmin=self.tmin, tmax=self.tmax,
                include_tmax=False).get_data().astype(np.float32))

    def _transform_event_id(self, raw):
        transform_dic = {
            '22': 'stimulus/memory/item_2',
            '44': 'stimulus/memory/item_4',
            '88': 'stimulus/memory/item_8',
            '122': 'stimulus/test/item_2',
            '144': 'stimulus/test/item_4',
            '188': 'stimulus/test/item_8',
            '1': 'response/correct',
            '3': 'response/wrong'
        }
        transform_event_id(raw, transform_dic)

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

    def _metadata_from_raw(self, raw):
        """...22---------------122-----------------1 or 3...
           ...|   maintenance   |   reaction time     |...
        """
        self._transform_event_id(raw)
        all_events, all_event_id = events_from_annotations(raw)
        row_events = [
            'stimulus/memory/item_2',
            'stimulus/memory/item_4',
            'stimulus/memory/item_8',
        ]
        keep_first = ['stimulus', 'test', 'response']
        metadata_tmin, metadata_tmax = 0.0, 5.5

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

    def _make_metadata(self, metadata):
        metadata['response_time'] = metadata['response'] - metadata['test']
        # add srt
        reaction_time_max = 1.617

        def srt(x):
            if x['first_response'] == 'correct':
                return reaction_time_max - x['response_time']
            else:
                return x['response_time'] - reaction_time_max

        metadata['srt'] = metadata.apply(srt, axis=1)
        metadata['srt_bins_freq'] = pd.qcut(metadata['srt'],
                                            3,
                                            labels=[0, 1, 2])
        bins = 3
        bins_edges = np.linspace(-reaction_time_max, reaction_time_max,
                                 bins + 1)
        metadata['srt_distribution_3'] = pd.cut(metadata['srt'],
                                                bins_edges,
                                                labels=[0, 1, 2])
        return metadata
