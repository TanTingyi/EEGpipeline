from mne.annotations import _annotations_starts_stops


def get_raw_edge_index(raw):
    """Returns the onset and ends index of the file splicing in
    the raw object. This method is very useful when using file 
    names to define metadata.

    Parameters
    ----------
    paths : mne.io.Raw
        Raw of one subject. 

    Returns         
    -------
    edge_index : dict
        Dict contain filename and it's onset & end.
        i.e. edge_index = {"s1_1.cnt":(0, 40), "s1_2.cnt":(40, 80)}
    
    """
    onsets, ends = _annotations_starts_stops(raw, 'edge', invert=True)
    edge_index = {}
    for path, onset, end in zip(raw.filenames, onsets, ends):
        edge_index[path] = (onset, end)
    return edge_index
