from mne import events_from_annotations, annotations_from_events


def transform_event_id(raw, transform_dic=None, description_transform=None):
    """Transform the description of Raw. 

    Parameters         
    ----------
    raw : mne.Raw
        Raw instance.
    transform_dic : None | dic
        Dictionary holds the new id required for conversion.
        Which key is the old id and the value is the new id.
    description_transform : None | callable
        Function use raw as input and return new_events and new_event_id.

    Returns
    -------
    None

    Notes
    -----
    """
    if description_transform:
        all_events, all_event_id = description_transform(raw)
    else:
        all_events, all_event_id = events_from_annotations(raw)

    if transform_dic:
        new_all_event_id = _transform_from_dict(all_event_id, transform_dic)
    else:
        new_all_event_id = {v: k for k, v in all_event_id.items()}

    annotation_new = annotations_from_events(all_events, raw.info['sfreq'],
                                             new_all_event_id)
    raw.set_annotations(annotation_new)


def _transform_from_dict(dic1, dic2):
    """Transform dic1's key from dic2's value. 
    """
    dic_new = {}
    for key, value in dic1.items():
        key_new = dic2[key] if key in dic2 else key
        dic_new[value] = key_new
    return dic_new
