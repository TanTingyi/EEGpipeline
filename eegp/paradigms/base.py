"""
Bast class of paradigm.
Authors: Matthew Tan <5636374@qq.com>
Time: 2021.4.15
"""
import abc
import os
from copy import deepcopy

import numpy as np

from .utils import pick_events_
from ..path import create_dir


class BaseParadigm(metaclass=abc.ABCMeta):
    """Abstract base class for paradigm-type classes.

    This class provides basic functionality and should 
    never be instantiated directly.
    """
    def __init__(self, code):
        """
        Parameters
        ----------
        code : string
            Unicode of paradigm.

        Attributes
        ----------
        self.raws : list
            List of Raw for each subject.
        self.epochs : list
            List of Epoch for each subject.
        self.paths : list
            List of FilePath for each subject.
        self.datas : list
            List of data for each subject.
        self.labels : list
            List of label for each subject.
        """
        self.code = code
        self._raws = []
        self._epochs = []
        self._paths = []
        self._datas = []
        self._labels = []

    @property
    def raws(self):
        return deepcopy(self._raws)

    @property
    def epochs(self):
        return deepcopy(self._epochs)

    @property
    def paths(self):
        return deepcopy(self._paths)

    @property
    def datas(self):
        return deepcopy(self._datas)

    @property
    def labels(self):
        return deepcopy(self._labels)

    @abc.abstractmethod
    def read_raw(self, paths):
        """Load raw file from FilePath. 
        
        Support to load only one subject or multiple subjects 
        at the same time. 
        

        Parameters
        ----------
        paths : FilePath | list
            FilePath that keep the load path and save path.

        Returns         
        -------
        None

        Notes
        -----
        Loading data of multiple subjects at the same time
        is suitable for the situation where you want to run ica
        to remove the eye by template matching method, 
        but should be careful that it will take up a lot of memory.
        """

        pass

    @abc.abstractmethod
    def preprocess(self):
        """Preprocess raw data. 

        Parameters
        ----------
        None

        Returns         
        -------
        None

        Notes
        -----
        You should implement preprocessing operations for 
        raw data in this method, such as bad channel marking 
        and repair, band-pass filtering and ICA to remove 
        EOG artifacts.

        Examples
        --------
        
        >>> for raw in self.__raws:
        >>>     raw = channel_repair_exclud(
        >>>         raw,
        >>>         exclude=['HEO', 'VEO'],
        >>>         montage='standard_1020')
        >>>     raw.filter(self.filter_low,
        >>>                self.filter_high,
        >>>                skip_by_annotation='edge')
        """
        pass

    @abc.abstractmethod
    def make_epochs(self):
        """Make epochs from raw. 

        Parameters
        ----------
        None

        Returns         
        -------
        None

        Notes
        -----
        You should split the raw data into epochs in this method, 
        and add meta information for the epochs if necessary.
        You can also implement the self._filter_epochs() method 
        to keep epochs you need.
        Examples
        --------
        >>> self.__epochs = []
        >>> for raw in self.__raws:
        >>>     events, event_id = self._define_trials(raw)
        >>>     epochs = Epochs(raw, events, event_id)
        >>>     epochs.metadata = self._metadata_from_raw(epochs, raw)
        >>>     epochs = self._filter_epochs(epochs)
        >>>     epochs.metadata = self._make_metadata(epochs.metadata)
        >>>     self.__epochs.append(epochs.resample(self.resample))

        """
        pass

    @abc.abstractmethod
    def pipeline(self, *args, **kwargs):
        """The main entrance of data processing.

        Notes
        -----
        This method should be user-defined.

        Examples
        --------
        
        >>> self.read_raw(filepaths)
        >>> self.preprocess()
        >>> self.make_epochs()
        >>> self.make_data()
        >>> self.save_data()
        """
        pass

    def make_data(self):
        """Make data and label.

        Parameters
        ----------
        None

        Returns         
        -------
        None

        Notes
        -----
        
        Should only be implemented in subclasses of the paradigm class, 
        used to define different types of labels in the same paradigm.

        """
        pass

    def save_raw(self, **kwargs):
        """Save raw file and return save path.

        Parameters
        ----------
        kwargs : kwargs for Raw.save().

        Returns         
        -------
        save_paths : list
            List containing the save path.

        Notes
        -----
        """
        save_paths = []
        for raw, path in zip(self.raws, self.paths):
            try:
                raw_path = os.path.join(path.save_path, 'raw')
            except TypeError:
                print('No save directory specified for subject {}!'.format(
                    path.subject))
            create_dir(raw_path)
            save_path = os.path.join(raw_path, path.subject) + '_raw.fif'
            raw.save(save_path, **kwargs)
            save_paths.append(save_path)
        return save_paths

    def save_epochs(self, **kwargs):
        """Save epochs file and return save path.

        Parameters
        ----------
        kwargs : kwargs for Epoch.save().

        Returns         
        -------
        save_paths : list
            List containing the save path.

        Notes
        -----
        """
        save_paths = []
        for epochs, path in zip(self.epochs, self.paths):
            try:
                epoch_path = os.path.join(path.save_path, 'epochs')
            except TypeError:
                print('No save directory specified for subject {}!'.format(
                    path.subject))
            create_dir(epoch_path)
            save_path = os.path.join(epoch_path, path.subject) + '_epo.fif'
            epochs.save(save_path, **kwargs)
            save_paths.append(save_path)

    def save_data(self):
        """Save data and label and return save path.

        Parameters
        ----------
        None 

        Returns         
        -------
        save_paths : list
            List containing the save path.

        Notes
        -----
        The saved file can be read by np.load()
        >>>    with np.load(file) as f:
        >>>        dataset = f['data']
        >>>        label = f['label']
        """
        save_paths = []
        for data, label, path in zip(self._datas, self._labels, self.paths):
            try:
                data_path = os.path.join(path.save_path, 'data')
            except TypeError:
                print('No save directory specified for subject {}!'.format(
                    path.subject))
            create_dir(data_path)
            save_path = os.path.join(data_path, path.subject)
            np.savez(save_path, data=data, label=label)
            save_paths.append(save_path)
        return save_paths

    def save_metadata(self, **kwargs):
        """Save metadata to .csv and return save path.

        Parameters
        ----------
        kwargs : kwargs for DataFrame.to_csv().

        Returns         
        -------
        save_paths : list
            List containing the save path.

        Notes
        -----
        """
        save_paths = []
        for epoch, path in zip(self.epochs, self.paths):
            try:
                metadata_path = os.path.join(path.save_path, 'metadata')
            except TypeError:
                print('No save directory specified for subject {}!'.format(
                    path.subject))
            create_dir(metadata_path)
            save_path = os.path.join(metadata_path, path.subject) + '.csv'
            epoch.metadata.to_csv(save_path, **kwargs)
            save_paths.append(save_path)
        return save_paths

    def _define_trials(self, raw, trial_ids):
        """Return new events and event_id for create epochs.
        
        Parameters         
        ----------
        raw : mne.Raw
            An instance of Raw.
        trial_ids : list
            All the id in the raw.annotation that represent the 
            beginning of the trial that you want to use.

        Returns
        -------
        events_new : array, shape (n_events, 3)
            The new defined events.
        event_id_new : dict
            The id of trials.

        Notes
        -----
        It is possible that annotation contains useless
        describe. In order to ignore irrelevant describe 
        when dividing trial, you need to specify the id that 
        represents the beginning of the trials.
        """
        return pick_events_(raw, trial_ids)

    def _metadata_from_raw(self, epochs, raw):
        """Return a DataFrame containing metadata.

        Add corresponding metadata for each epoch in epochs.
        
        Parameters         
        ----------
        epochs : mne.Epochs
            Epochs instance made in self.make_epochs()
        raw : mne.Raw
            An instance of Raw.

        Returns
        -------
        metadata : DataFrame
            Metadata corresponding to epochs.

        Notes
        -----
        Metadata is usually made based on the information 
        contained in the raw file, such as the type of experimental 
        stimulus represented by the file name, the subject responds time, 
        the accuracy rate, etc.
        """
        pass

    def _make_metadata(self, metadata):
        """Build and return new metadata based on existing metadata.

        Create high-level metadata based on metadata extracted from raw files
        
        Parameters         
        ----------
        metadata : DataFrame
            Metadata corresponding to epochs.

        Returns
        -------
        metadata : DataFrame
            New metadata that contains more information.

        Notes
        -----
        Metadata extracted from raw data may not be enough. 
        Therefore, it is necessary to construct high-level metadata 
        based on existing metadata. Such as constructing SRT scores 
        based on the subject's response time and accuracy.
        """
        return metadata

    def _filter_epochs(self, epochs):
        """Return the epochs you want to keep.
        
        Parameters         
        ----------
        epochs : mne.Epochs
            Epochs instance made in self.make_epochs()

        Returns
        -------
        epochs : mne.Epochs
            Epochs that meet the rules.

        Notes
        -----
        """
        return epochs
