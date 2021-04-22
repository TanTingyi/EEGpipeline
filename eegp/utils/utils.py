import re
import os

import mne
import numpy as np
import scipy.io as scio

from os.path import isfile
from pathlib import Path

from mne.io import read_raw_bdf
from mne import Annotations
from mne.annotations import _annotations_starts_stops


def _read_annotations_bdf(annotations):
    pat = '([+-]\\d+\\.?\\d*)(\x15(\\d+\\.?\\d*))?(\x14.*?)\x14\x00'
    if isinstance(annotations, str):
        with open(annotations, encoding='latin-1') as annot_file:
            triggers = re.findall(pat, annot_file.read())
    else:
        tals = bytearray()
        for chan in annotations:
            this_chan = chan.ravel()
            if this_chan.dtype == np.int32:  # BDF
                this_chan.dtype = np.uint8
                this_chan = this_chan.reshape(-1, 4)
                # Why only keep the first 3 bytes as BDF values
                # are stored with 24 bits (not 32)
                this_chan = this_chan[:, :3].ravel()
                for s in this_chan:
                    tals.extend(s)
            else:
                for s in this_chan:
                    i = int(s)
                    tals.extend(np.uint8([i % 256, i // 256]))

        # use of latin-1 because characters are only encoded for the first 256
        # code points and utf-8 can triggers an "invalid continuation byte"
        # error
        triggers = re.findall(pat, tals.decode('latin-1'))

    events = []
    for ev in triggers:
        onset = float(ev[0])
        duration = float(ev[2]) if ev[2] else 0
        for description in ev[3].split('\x14')[1:]:
            if description:
                events.append([onset, duration, description])
    return zip(*events) if events else (list(), list(), list())


def read_mat(filepath):
    mat = scio.loadmat(filepath)
    ch_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
        'F8', 'T3', 'T4', 'T5', 'T6'
    ]
    sfreq = 1000
    data = mat['eegdata'][0][0][1] * 1e-6
    onset = mat['eegdata'][0][0][4].squeeze()
    onset = [sample / sfreq for sample in onset]
    duration = np.zeros_like(onset)
    description = np.array(['onset'] * len(onset))

    info = mne.create_info(ch_names, sfreq, 'eeg')
    raw = mne.io.RawArray(data, info)
    annotations = mne.Annotations(onset, duration, description)
    raw.set_annotations(annotations)
    return raw


def read_raw_brk(filepath, **kwargs):
    raw = read_raw_bdf(filepath, **kwargs)
    dic, _ = os.path.split(filepath)
    annotation = read_raw_bdf(os.path.join(dic, 'evt.bdf'), **kwargs)

    idx = np.empty(0, int)
    tal_data = annotation._read_segment_file(np.empty(
        (0, annotation.n_times)), idx, 0, 0, int(annotation.n_times),
                                             np.ones((len(idx), 1)), None)
    onset, duration, description = _read_annotations_bdf(tal_data[0])
    onset = np.array([i for i in onset], dtype=np.float64)
    duration = np.array([int(i) for i in duration], dtype=np.int64)
    desc = np.array([int(i) for i in description], dtype=np.int64)
    annotation_new = Annotations(onset, duration, desc)
    raw.set_annotations(annotation_new)
    return raw


def find_path_not_exist(paths):
    """Recursively check whether the path in the list exists.

    takes a list as input and return path if it is not exist.

    Parameters
    ----------
    paths : list.
        list containing paths.

    Return
    ------
    paths_not_exist : list
        paths which not exist.
    """
    paths_not_exist = []

    def _check_file(paths):
        if not isinstance(paths, list):
            if not isfile(paths):
                return paths
        else:
            for path in paths:
                ans = _check_file(path)
                if isinstance(ans, str):
                    paths_not_exist.append(ans)

    _check_file(paths)
    return paths_not_exist


def depth_count(x):
    """Return maximum depth of the returned list.

    Parameters
    ----------
    x : list.

    Return
    ------
    ans : int
        maximum depth.
    """
    return int(isinstance(x, list)) and len(x) and 1 + max(map(depth_count, x))


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _check_matches(arr1, arr2):
    # check matches of two array
    if len(arr1) != len(arr2):
        return False
    for data1, data2 in zip(arr1, arr2):
        if data1 != data2:
            return False
    return True


def _edge_index(raw):
    onsets, ends = _annotations_starts_stops(raw, 'edge', invert=True)
    index = {}
    for path, onset, end in zip(raw.filenames, onsets, ends):
        index[path] = (onset, end)
    return index
