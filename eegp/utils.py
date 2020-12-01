import re
import numpy as np
import mne
import scipy.io as scio


def read_annotations_bdf(annotations):
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


def read_mat(path):
    mat = scio.loadmat(path)
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
