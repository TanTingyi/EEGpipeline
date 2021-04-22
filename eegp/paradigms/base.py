"""
Bast class of paradigm.
Authors: Matthew Tan <5636374@qq.com>
Time: 2021.4.15
"""
import abc
import os

import numpy as np

from ..utils import create_dir


class BaseParadigm(metaclass=abc.ABCMeta):
    """Abstract base class for paradigm-type classes.

    This class provides basic functionality and should never be instantiated
    directly.
    """
    def __init__(self, code):
        """
        Parameters
        ----------
        code: string
            数据集的唯一标号
        """
        self.code = code

    @abc.abstractmethod
    def read_raw(self, paths):
        """Load raw file.

        Parameters         
        ----------
        paths : FilePath | list            
        
        Returns         
        -------
        None
        """
        pass

    @abc.abstractmethod
    def preprocess(self):
        """推荐的预处理流程：
            坏道插值 -> 带通滤波 -> ICA 去眼电 -> 重参考

        """
        pass

    @abc.abstractmethod
    def make_epochs(self):
        """Make epochs by trial id.
        """
        pass

    @abc.abstractmethod
    def pipeline(self, *args, **kwargs):
        """数据处理 pipeline，定义数据的处理流程
        """
        pass

    def make_data(self):
        """Make data and label by rules.
        """
        pass

    def save_raw(self, **kwargs):
        for raw, path in zip(self.raws, self.paths):
            raw_path = os.path.join(path.save_path, 'raw')
            create_dir(raw_path)
            raw.save(
                os.path.join(raw_path, path.subject) + '_raw.fif', **kwargs)

    def save_epochs(self, **kwargs):
        for epochs, path in zip(self.epochs, self.paths):
            epoch_path = os.path.join(path.save_path, 'epochs')
            create_dir(epoch_path)
            epochs.save(
                os.path.join(epoch_path, path.subject) + '_epo.fif', **kwargs)

    def save_data(self, **kwargs):
        for data, label, path in zip(self._datas, self._labels, self.paths):
            data_path = os.path.join(path.save_path, 'data')
            create_dir(data_path)
            np.savez(os.path.join(data_path, path.subject),
                     data=data,
                     label=label)

    def save_metadata(self, **kwargs):
        for epoch, path in zip(self.epochs, self.paths):
            metadata_path = os.path.join(path.save_path, 'metadata')
            create_dir(metadata_path)
            epoch.metadata.to_csv(
                os.path.join(metadata_path, path.subject) + '.csv', **kwargs)

    @abc.abstractmethod
    def _define_trials(self, raw):
        """Define new events only contain trial start event.

        Parameters         
        ----------
        raw : mne.Raw
            An instance of Raw.
             
        Returns
        -------
        events_new : array, shape (n_events, 3)
            The new defined events.
        event_id_new : dict
            The id of trials.
        """
        pass

    def _metadata_from_raw(self, epochs, raw):
        pass

    def _make_metadata(self, metadata):
        return metadata

    def _filter_epochs(self, epochs):
        return epochs
