import numpy as np
from mne.datasets import eegbci
from eegp.paradigms import PhysioNetMI
from eegp.path import FilePath


def make_filepath(dir_save, subs):
    filepaths = []
    for i in subs:
        load_path = eegbci.load_data(int(i), [4, 6, 8, 10, 12, 14])
        filepaths.append(
            FilePath(subject='s{}'.format(int(i)),
                     filetype='edf',
                     load_path=load_path,
                     save_path=dir_save))
    return filepaths


class RemoveEOG(PhysioNetMI):
    def pipeline(self, filepaths):
        self.read_raw(filepaths)
        self.preprocess()
        self.make_epochs()
        self.make_data()
        self.save_data()


if __name__ == "__main__":
    rm_group_eog = RemoveEOG(tmin=0.,
                             tmax=1.,
                             baseline=None,
                             filter_low=0.5,
                             filter_high=45.,
                             resample=128,
                             remove_eog=True)
    group_size = 1
    sub_group = np.array_split(np.arange(1, 6), group_size)
    for subs in sub_group:
        filepaths = make_filepath(r'E:\Data\tmp\mi', subs)
        rm_group_eog.pipeline(filepaths)
