import os
import mne
import numpy as np

from mne.channels import make_standard_montage
from mne.io import concatenate_raws
from mne import Epochs
from mne import pick_events

from .base import BaseDataset


class STEW(BaseDataset):
    """STEW的基类
    """
    def __init__(self,
                 code,
                 tmin=0,
                 tmax=4,
                 step=1,
                 filter_low=0.5,
                 filter_high=None,
                 resample=128,
                 baseline=None,
                 reject=None):
        """
        Parameters
        ----------
        tmin : int | float
            提取数据的开始时间点
        tmax : int | float
            提取数据的结束时间点
        step : int | float             
            提取数据的滑动窗口步长
        filter_low : None | int | float
            带通滤波器的下截止频率
        filter_high : None | int | float
            带通滤波器的上截止频率
        resample : int
            重采样的采样频率
        baseline : None | tuple
            基线矫正的时间段，如果是 None 则不做基线矫正
        reject : None | dict
            阈值去除的阈值，例如 dict(eeg=200e-6) 将在
            分 trail 的时候丢弃峰峰值为 200 mv 的数据段
            为 None 时不启用
        """
        super(STEW, self).__init__(code=code)
        self.tmin = tmin
        self.tmax = tmax
        self.step = step
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.resample = resample
        self.baseline = baseline
        self.reject = reject

    def _loadtxt(self, path):
        data = np.loadtxt(path)
        data = data.T * 1e-6
        ch_names = [
            'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8',
            'FC6', 'F4', 'F8', 'AF4'
        ]
        info = mne.create_info(ch_names, self.resample, 'eeg')
        raw = mne.io.RawArray(data, info)
        return raw

    def get_raw(self, path, bad_path=None):
        if isinstance(path, str):
            path = [path]
        elif isinstance(path, list):
            if not isinstance(path[0], str):
                raise ValueError('输入必须是路径的字符串或者包含路径的列表')
        else:
            raise ValueError('输入必须是路径的字符串或者包含路径的列表')

        raws = []
        for f in path:
            try:
                raws.append(self._loadtxt(f))
            except FileNotFoundError:
                continue

        if len(raws) == 0:
            return

        raw = concatenate_raws(raws)

        if bad_path:
            with open(bad_path, 'r') as f:
                bad_channel = f.read().split()
            raw.info['bads'] = bad_channel

        return raw

    def preprocess(self, raws):
        if not isinstance(raws, list):
            raws = [raws]
        for raw in raws:
            # 去除部分无关电极并对坏道插值
            raw = self._run_channel_repair_exclud(raw)

            # 带通滤波
            raw.filter(self.filter_low,
                       self.filter_high,
                       skip_by_annotation='edge')

        return raws[0] if len(raws) == 1 else raws

    def get_epochs(self, raw):
        events, event_id = self._make_event(raw)
        epochs = Epochs(raw,
                        events,
                        event_id,
                        self.tmin - 0.2,
                        self.tmax + 0.2,
                        proj=True,
                        baseline=self.baseline,
                        reject=self.reject,
                        preload=True)
        return epochs

    def _run_reference(self, raw):
        # REST 重参考
        sphere = mne.make_sphere_model('auto', 'auto', raw.info)
        src = mne.setup_volume_source_space(sphere=sphere,
                                            exclude=30.,
                                            pos=15.)
        forward = mne.make_forward_solution(raw.info,
                                            trans=None,
                                            src=src,
                                            bem=sphere)
        raw.set_eeg_reference('REST', forward=forward)
        return raw

    def _run_channel_repair_exclud(self, raw):
        raw.pick(picks='eeg')
        ten_twenty_montage = make_standard_montage('standard_1020')
        raw.set_montage(ten_twenty_montage, match_case=False)
        raw.interpolate_bads()
        return raw

    def get_data(self, epochs):
        data = epochs.copy().crop(tmin=self.tmin,
                                  tmax=self.tmax,
                                  include_tmax=False).get_data().astype(
                                      np.float32)
        label = epochs.events[:, 2]
        return data, label

    def _make_event(self, raw):
        pass

    def _make_window(self, raw):
        window_len = self.tmax * self.resample
        step = self.step * self.resample
        head = 0
        tail = head + window_len
        window_index = []
        while tail <= raw.n_times:
            window_index.append(head)
            head += step
            tail += step
        return window_index


class NoTask(STEW):
    def __init__(self, *args, **kwargs):
        super(NoTask, self).__init__(code='No Task', *args, **kwargs)

    def _make_event(self, raw):
        window_index = self._make_window(raw)
        events = np.array([[i, 0, 0] for i in window_index])
        event_id = {'notask': 0}
        return events, event_id


class STask(STEW):
    def __init__(self, *args, **kwargs):
        super(STask, self).__init__(code='SIMKAP Task', *args, **kwargs)

    def _make_event(self, raw):
        window_index = self._make_window(raw)
        events = np.array([[i, 0, 10] for i in window_index])
        event_id = {'task': 10}
        return events, event_id


class Rating(STEW):
    def __init__(self, rate, *args, **kwargs):
        super(Rating, self).__init__(code='Rating Task', *args, **kwargs)
        self.rate = rate

    def _make_event(self, raw):
        window_index = self._make_window(raw)
        events = np.array([[i, 0, self.rate] for i in window_index])
        if 1 <= self.rate < 4:
            key = 'lo'
        elif 4 <= self.rate < 7:
            key = 'mi'
        elif 7 <= self.rate < 10:
            key = 'hi'
        event_id = {key:self.rate}
        return events, event_id
