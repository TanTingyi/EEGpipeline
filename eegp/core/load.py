import re
import os

import mne
import numpy as np
import scipy.io as scio

from mne import Annotations
from mne.io import concatenate_raws, read_raw_cnt, read_raw_edf, read_raw_bdf


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


def read_raw_brk(path, **kwargs):
    """Read function for Neuracle Technology.

    Parameters
    ----------
    path : str
        Path to the data.bdf.

    Returns         
    -------
    raw : instance of RawBDF.
        The raw data.

    Notes
    -----
    data.bdf and evt.bdf must be in the same path.
    """
    raw = read_raw_bdf(path, **kwargs)
    dic, _ = os.path.split(path)
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


def _read_mat(filepath):
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
    annotations = Annotations(onset, duration, description)
    raw.set_annotations(annotations)
    return raw


def _read_func(filetype, filepath):
    if filetype == 'cnt':
        return read_raw_cnt(filepath, preload=True)
    elif filetype == 'edf':
        return read_raw_edf(filepath, preload=True)
    elif filetype == 'bdf':
        return read_raw_bdf(filepath, preload=True)
    elif filetype == 'brk':
        return read_raw_brk(filepath, preload=True)


def read_raw(paths):
    """Return raws list from paths. 

    Parameters
    ----------
    paths : list of FilePath.
        List includes the FilePath of all subjects.

    Returns         
    -------
    raws : list.
        List includes the raw of all subjects.

    Notes
    -----
    """
    raws = []
    for subject_path in paths:
        raw = concatenate_raws([
            _read_func(subject_path.filetype, f)
            for f in subject_path.load_path
        ])
        if subject_path.bad_channel_path:
            with open(subject_path.bad_channel_path, 'r') as f:
                bad_channel = f.read().split()
            raw.info['bads'] = bad_channel
        raws.append(raw)
    return raws
