import os

import numpy as np

from eegp.paradigms import OneBack
from eegp.path import FilePath


def get_files_by_runs(dir, runs):
    """
    """
    file_paths = [os.path.join(dir, str(run)) for run in runs]
    file_paths = [os.path.join(path, 'data.bdf') for path in file_paths]
    return file_paths


def make_filepath(dir_load, dir_save, subs, runs):
    filepaths = []
    for i in subs:
        dir_load_sub = os.path.join(dir_load, 'S' + str(i))
        filepaths.append(
            FilePath(subject='S{}'.format(int(i)),
                     filetype='brk',
                     load_path=get_files_by_runs(dir_load_sub, runs),
                     save_path=dir_save))
    return filepaths


class Test(OneBack):
    def pipeline(self, filepaths):
        self.read_raw(filepaths)
        self.preprocess()
        self.make_epochs()
        self.make_data()
        self.save_data()
        self.save_metadata()


if __name__ == "__main__":
    # TODO 仔细看一些打标错误trial的数据
    dir_load = r'E:\Data'
    dir_save = r'E:\Data\tmp\nback'
    group_size = 1
    sub_group = np.array_split(np.arange(1, 2), group_size)
    runs = [1, 2, 3, 4, 5, 6]
    params = dict(tmin=0.,
                  tmax=1.,
                  baseline=None,
                  reject=None,
                  filter_low=0.5,
                  filter_high=None,
                  resample=250,
                  remove_eog=False)
    for subs in sub_group:
        filepaths = make_filepath(dir_load, dir_save, subs, runs)
        srt = Test(**params)
        srt.pipeline(filepaths)
