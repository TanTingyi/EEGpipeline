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

if __name__ == "__main__":
    dm = RemoveEOG(tmin=1.,
                   tmax=2,
                   baseline=None,
                   reject=None,
                   filter_low=0.5,
                   filter_high=45.,
                   resample=160,
                   verbose=0)
    path = r's1_raw.fif'
    dm.pipeline(path)
