from mne import make_sphere_model, setup_volume_source_space, make_forward_solution
from mne.channels import make_standard_montage
from mne.preprocessing import ICA, corrmap


def channel_repair_exclud(raws, exclude, montage):
    """Remove unnecessary channels and repair bad sectors.

    Parameters
    ----------
    raws : list of Raw.
        The raw data.
    exclude : list
        List of channel name.
    montage : str
        Multiple channel montage supported by MNE.
        
    Returns         
    -------
    raws : list of Raw.
        The raw data.

    Notes
    -----
    """
    if not isinstance(raws, list):
        raws = [raws]
    for raw in raws:
        raw.pick(picks='eeg', exclude=exclude)
        raw.set_montage(make_standard_montage(montage), match_case=False)
        raw.interpolate_bads()
    return raws


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
    ica = ICA(n_components=n_components, max_iter='auto')
    ica.fit(raw, verbose=0)
    while threshold > 1:
        eog_inds, _ = ica.find_bads_eog(raw,
                                        ch_name=ch_name,
                                        threshold=threshold,
                                        verbose=0)
        if eog_inds:
            break
        threshold -= 0.3

    if not eog_inds:
        raise RuntimeError('Didn\'t find a EOG component.')

    ica.plot_properties(raw, eog_inds)
    ica.exclude = eog_inds
    ica.apply(raw, verbose=0)
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
    icas = [
        ICA(n_components=n_components, max_iter='auto').copy().fit(raw,
                                                                   verbose=0)
        for raw in raws
    ]
    for raw, ica in zip(raws, icas):
        eog_inds, _ = ica.find_bads_eog(raw, ch_name=ch_name, verbose=0)
        if eog_inds:
            break
    if not eog_inds:
        raise RuntimeError(
            'This group of subjects does not have an EOG template, add more subjects.'
        )

    _ = corrmap(icas,
                template=(0, eog_inds[0]),
                threshold=threshold,
                label='blink')
    for raw, ica in zip(raws, icas):
        ica.exclude = ica.labels_['blink']
        ica.apply(raw, verbose=0)
    return raws


def rest_reference(raws):
    """Re-reference the data by REST.

    Parameters
    ----------
    raws : list of Raw.
        The raw data.
        
    Returns         
    -------
    raws : list of Raw.
        The raw data.

    Notes
    -----
    """
    if not isinstance(raws, list):
        raws = [raws]
    for raw in raws:
        sphere = make_sphere_model('auto', 'auto', raw.info, verbose=0)
        src = setup_volume_source_space(sphere=sphere,
                                        exclude=30.,
                                        pos=15.,
                                        verbose=0)
        forward = make_forward_solution(raw.info,
                                        trans=None,
                                        src=src,
                                        bem=sphere,
                                        verbose=0)
        raw.set_eeg_reference('REST', forward=forward)
    return raws


def band_pass_filter(raws, low, high):
    """Bandpass filter.

    Parameters
    ----------
    raws : list of Raw.
        The raw data.
        
    Returns         
    -------
    raws : list of Raw.
        The raw data.

    Notes
    -----
    """
    if not isinstance(raws, list):
        raws = [raws]
    for raw in raws:
        raw.filter(low,
                   high,
                   skip_by_annotation='edge',
                   method='iir',
                   verbose=0)
    return raws
