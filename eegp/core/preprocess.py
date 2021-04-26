from mne import make_sphere_model, setup_volume_source_space, make_forward_solution
from mne.channels import make_standard_montage
from mne.preprocessing import ICA, corrmap


def channel_repair_exclud(raw, exclude, montage):
    """Remove unnecessary channels and repair bad sectors.

    Parameters
    ----------
    raw : instance of Raw.
        The raw data.
    exclude : list
        List of channel name.
    montage : str
        Multiple channel montage supported by MNE.

    Returns         
    -------
    raw : instance of Raw.
        The raw data.

    Notes
    -----
    """
    raw.pick(picks='eeg', exclude=exclude)
    raw.set_montage(make_standard_montage(montage), match_case=False)
    raw.interpolate_bads()
    return raw


def remove_eog_ica(raw, n_components, ch_name, threshold):
    """Remove EOG artifacts by ICA.

    Parameters
    ----------
    raw : instance of Raw.
        The raw data.
    n_components : int
        Number of principal components for ICA.
    ch_name : str
        The name of the channel to use for EOG peak detection.
    threshold : int
        The value above which a feature is classified as outlier.

    Returns         
    -------
    raw : instance of Raw.
        The raw data.

    Notes
    -----
    """
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
    """Remove EOG artifacts by similar Independent Components across subjects.

    Parameters
    ----------
    raws : list
        List Raw instances.
    n_components : int
        Number of principal components for ICA.
    ch_name : str
        The name of the channel to use for EOG peak detection.
    threshold : int
        The value above which a feature is classified as outlier.

    Returns         
    -------
    raws : list
        List Raw instances.

    Notes
    -----
    """
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
    """Re-reference the data by REST.

    Parameters
    ----------
    raw : instance of Raw.
        The raw data.
        
    Returns         
    -------
    raw : instance of Raw.
        The raw data.

    Notes
    -----
    """
    sphere = make_sphere_model('auto', 'auto', raw.info)
    src = setup_volume_source_space(sphere=sphere, exclude=30., pos=15.)
    forward = make_forward_solution(raw.info, trans=None, src=src, bem=sphere)
    raw.set_eeg_reference('REST', forward=forward)
    return raw
