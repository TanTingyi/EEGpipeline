from mne.event import pick_events
from mne import events_from_annotations


def pick_events_(raw, target_ids):
    events_raw, event_id_raw = events_from_annotations(raw)
    event_id_new = {
        key: event_id_raw[key]
        for key in event_id_raw if key in target_ids
    }
    events_new = pick_events(events_raw, include=list(event_id_new.values()))
    return events_new, event_id_new
