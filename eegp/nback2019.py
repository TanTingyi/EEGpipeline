import os
import mne
import numpy as np

from mne.channels import make_standard_montage
from mne.io import concatenate_raws
from mne import Epochs, Annotations, events_from_annotations
from mne import pick_events

from .base import BaseDataset
from .utils import read_mat


class NBack(BaseDataset):
    """NBack的基类
    """
    def __init__(self,
                 code,
                 tmin=0,
                 tmax=1,
                 filter_low=0.5,
                 filter_high=None,
                 resample=160,
                 baseline=None,
                 reject=None):
        """
        Parameters
        ----------
        tmin : int | float
            提取数据的开始时间点
        tmax : int | float
            提取数据的结束时间点
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
        super(NBack, self).__init__(code=code)
        self.tmin = tmin
        self.tmax = tmax
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.resample = resample
        self.baseline = baseline
        self.reject = reject

    def get_raw(self, paths, bad_path=None):
        if isinstance(paths, str):
            paths = [paths]
        elif isinstance(paths, list):
            if not isinstance(paths[0], str):
                raise ValueError('输入必须是路径的字符串或者包含路径的列表')
        else:
            raise ValueError('输入必须是路径的字符串或者包含路径的列表')

        raws = []
        for path in paths:
            try:
                raws.append(read_mat(path))
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
        events_old, event_id_old = events_from_annotations(raw)
        events_new, event_id_new = self._event_tranform(
            events_old, event_id_old)
        epochs = Epochs(raw,
                        events_new,
                        event_id_new,
                        self.tmin - 0.2,
                        self.tmax + 0.2,
                        proj=True,
                        baseline=self.baseline,
                        reject=self.reject,
                        preload=True)
        epochs = epochs.resample(self.resample)
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
        raw.pick(picks='eeg', exclude=['HEO', 'VEO', 'M1', 'M2'])
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


class OneBack(NBack):
    def __init__(self, *args, **kwargs):
        super(OneBack, self).__init__(code='1-back', *args, **kwargs)

    def _event_tranform(self, events_old, event_id_old):
        '''10 11 代表刺激出现
        其他的代表被试按键的 ascii
        添加当前 epochs 是 1back 的标签，并去除其他标签
        '''
        event_id_new = {'one': 0}
        events_new = pick_events(events_old, include=[event_id_old['onset']])

        events_new[:, -1][events_new[:, -1] ==
                          event_id_old['onset']] = event_id_new['one']

        return events_new, event_id_new


class TwoBack(NBack):
    def __init__(self, *args, **kwargs):
        super(TwoBack, self).__init__(code='2-back', *args, **kwargs)

    def _event_tranform(self, events_old, event_id_old):
        '''10 11 代表刺激出现
        其他的代表被试按键的 ascii
        添加当前 epochs 是 2back 的标签，并去除其他按键标签
        '''
        event_id_new = {'two': 1}
        events_new = pick_events(events_old, include=[event_id_old['onset']])

        events_new[:, -1][events_new[:, -1] ==
                          event_id_old['onset']] = event_id_new['two']

        return events_new, event_id_new
