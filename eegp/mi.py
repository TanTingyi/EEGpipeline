import os

import numpy as np
import mne

from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, events_from_annotations
from mne.preprocessing import ICA, corrmap
from mne.datasets import eegbci

from .base import BaseDataset


class MI(BaseDataset):
    """PhysioNet 运动想象
    """
    def __init__(self,
                 tmin,
                 tmax,
                 filter_low=7.,
                 filter_high=30.,
                 resample=160,
                 baseline=None,
                 reject=None,
                 verbose=0):
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
        verbose : int | bool
            数据处理过程中是否打印状态
        """
        super(MI, self).__init__(code='PhysioNet MI')
        self.tmin = tmin
        self.tmax = tmax
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.resample = resample
        self.baseline = baseline
        self.reject = reject
        self.verbose = verbose

    def get_raw(self, paths, bad_path=None):
        raw = concatenate_raws([
            read_raw_edf(f, preload=True, verbose=self.verbose) for f in paths
        ])
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
                       skip_by_annotation='edge',
                       verbose=self.verbose)
        # ICA 去眼电
        if len(raws) > 1:
            raws = self._run_template_ica(raws)
        else:
            raws[0] = self._run_ica(raws[0])

        for raw in raws:
            # REST 重参考
            raw = self._run_reference(raw)

        return raws[0] if len(raws) == 1 else raws

    def get_epochs(self, raw):
        events, _ = events_from_annotations(raw,
                                            event_id=dict(T1=2, T2=3),
                                            verbose=self.verbose)
        event_id = dict(hands=2, feet=3)

        epochs = Epochs(raw,
                        events,
                        event_id,
                        self.tmin - 0.2,
                        self.tmax + 0.2,
                        proj=True,
                        baseline=self.baseline,
                        reject=self.reject,
                        preload=True,
                        verbose=self.verbose)
        epochs = epochs.resample(self.resample)
        return epochs

    def get_data(self, epochs):
        data = epochs.crop(tmin=self.tmin,
                           tmax=self.tmax,
                           verbose=self.verbose).get_data().astype(np.float32)
        label = epochs.events[:, -1]
        return data, label

    def _run_ica(self, raw):
        ica = ICA(n_components=15, random_state=2020)
        ica.fit(raw, verbose=self.verbose)
        eog_inds, _ = ica.find_bads_eog(raw,
                                        ch_name='Fpz',
                                        threshold=3,
                                        verbose=self.verbose)
        if not eog_inds:
            raise RuntimeError('未找到合适眼电成分，减小阈值继续尝试')
        ica.plot_properties(raw, eog_inds)
        ica.exclude = eog_inds
        ica.apply(raw)
        return raw

    def _run_template_ica(self, raws):
        icas = [
            ICA(n_components=15).copy().fit(raw, verbose=self.verbose)
            for raw in raws
        ]
        for raw, ica in zip(raws, icas):
            eog_inds, _ = ica.find_bads_eog(raw,
                                            ch_name='Fpz',
                                            verbose=self.verbose)
            if eog_inds:
                break
        if not eog_inds:
            raise RuntimeError('本组被试没有眼电相关模板，增加组内被试')

        # 使用模板匹配，基于某一个被试的 EOG 相关的 IC，计算其他被试的 IC
        _ = corrmap(icas,
                    template=(0, eog_inds[0]),
                    threshold=0.8,
                    label='blink',
                    verbose=self.verbose)
        for raw, ica in zip(raws, icas):
            ica.exclude = ica.labels_['blink']
            ica.apply(raw)
        return raws

    def _run_reference(self, raw):
        # REST 重参考
        sphere = mne.make_sphere_model('auto',
                                       'auto',
                                       raw.info,
                                       verbose=self.verbose)
        src = mne.setup_volume_source_space(sphere=sphere,
                                            exclude=30.,
                                            pos=15.,
                                            verbose=self.verbose)
        forward = mne.make_forward_solution(raw.info,
                                            trans=None,
                                            src=src,
                                            bem=sphere,
                                            verbose=self.verbose)
        raw.set_eeg_reference('REST', forward=forward, verbose=self.verbose)
        return raw

    def _run_channel_repair_exclud(self, raw):
        eegbci.standardize(raw)  # set channel names
        montage = make_standard_montage('standard_1005')
        raw.set_montage(montage)
        # strip channel names of "." characters
        raw.rename_channels(lambda x: x.strip('.'))
        raw.pick(picks='eeg', exclude='bads')
        return raw
