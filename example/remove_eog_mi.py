import os

from mne.datasets import eegbci
from eegp.mi import MI


class RemoveEOG(MI):
    def pipeline(self, path):
        """对单个被试使用ICA去除眼电，并保存结果
        """
        subject = 1
        runs = [6, 10, 14]  # motor imagery: hands vs feet
        raw_fnames = eegbci.load_data(subject, runs)

        raw = self.get_raw(raw_fnames)
        raw = self.preprocess(raw)
        raw.save(path)


class RemoveGroupEOG(MI):
    def pipeline(self, save_path):
        """使用模板匹配的方法去除眼电，并保存结果
        """
        subjects = range(1, 5)
        runs = [6, 10, 14]  # motor imagery: hands vs feet
        paths = [eegbci.load_data(subject, runs) for subject in subjects]
        raws = [self.get_raw(path) for path in paths]

        raws = self.preprocess(raws)
        for i, raw in enumerate(raws):
            raw.save(os.path.join(save_path, str(i + 1) + '_raw.fif'))

if __name__ == "__main__":
    rm_group_eog = RemoveGroupEOG(tmin=1.,
                                  tmax=2,
                                  baseline=None,
                                  reject=None,
                                  filter_low=0.5,
                                  filter_high=45.,
                                  resample=160,
                                  verbose=0)
    path = r'/content'
    rm_group_eog.pipeline(path)
