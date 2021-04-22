import os
import numpy as np

from mne.channels import make_standard_montage
from mne.preprocessing import ICA, corrmap
from mne.io import concatenate_raws, read_raw_cnt, read_raw_edf, read_raw_bdf
from mne import make_sphere_model, setup_volume_source_space, make_forward_solution
from ..utils import find_path_not_exist, read_raw_brk


def read_func(filetype, filepath):
    if filetype == 'cnt':
        return read_raw_cnt(filepath, preload=True)
    elif filetype == 'edf':
        return read_raw_edf(filepath, preload=True)
    elif filetype == 'bdf':
        return read_raw_bdf(filepath, preload=True)
    elif filetype == 'brk':
        return read_raw_brk(filepath, preload=True)


def read_raw(paths):
    raws = []
    for subject_path in paths:
        raw = concatenate_raws([
            read_func(subject_path.filetype, f) for f in subject_path.load_path
        ])
        if subject_path.bad_channel_path:
            with open(subject_path.bad_channel_path, 'r') as f:
                bad_channel = f.read().split()
            raw.info['bads'] = bad_channel
        raws.append(raw)
    return raws


def channel_repair_exclud(raw, exclude, montage):
    raw.pick(picks='eeg', exclude=exclude)
    ten_twenty_montage = make_standard_montage(montage)
    raw.set_montage(ten_twenty_montage, match_case=False)
    raw.interpolate_bads()
    return raw


def remove_eog_ica(raw, n_components, ch_name, threshold):
    ica = ICA(n_components=n_components)
    ica.fit(raw)
    eog_inds, _ = ica.find_bads_eog(raw, ch_name=ch_name, threshold=threshold)
    if not eog_inds:
        raise RuntimeError('未找到合适眼电成分，减小阈值继续尝试')
    ica.plot_properties(raw, eog_inds)
    ica.exclude = eog_inds
    ica.apply(raw)
    return raw


def remove_eog_template_ica(raws, n_components, ch_name, threshold):
    icas = [ICA(n_components=n_components).copy().fit(raw) for raw in raws]
    for raw, ica in zip(raws, icas):
        eog_inds, _ = ica.find_bads_eog(raw, ch_name=ch_name)
        if eog_inds:
            break
    if not eog_inds:
        raise RuntimeError(
            'This group of subjects does not have an EOG template, add more subjects'
        )

    _ = corrmap(icas,
                template=(0, eog_inds[0]),
                threshold=threshold,
                label='blink')
    for raw, ica in zip(raws, icas):
        ica.exclude = ica.labels_['blink']
        ica.apply(raw)
    return raws


def rest_reference(raw):
    # REST reference
    sphere = make_sphere_model('auto', 'auto', raw.info)
    src = setup_volume_source_space(sphere=sphere, exclude=30., pos=15.)
    forward = make_forward_solution(raw.info, trans=None, src=src, bem=sphere)
    raw.set_eeg_reference('REST', forward=forward)
    return raw
