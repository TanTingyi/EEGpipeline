import os

import numpy as np

from eegp.paradigms import LetterDelayMatch
from eegp.path import FilePath


def get_files_in_folder(dir):
    """返回文件夹下面所有以 s 开头，并且以 cnt 结尾的文件路径
    """
    file_paths = list(
        filter(lambda x: x.endswith('cnt') and x.startswith('s'),
               os.listdir(dir)))
    file_paths = [os.path.join(dir, path) for path in file_paths]
    return file_paths


def make_filepath(dir_load, dir_save, subs):
    filepaths = []
    for i in subs:
        dir_load_sub = os.path.join(dir_load, 's' + str(i))
        filepaths.append(
            FilePath(subject='s{}'.format(int(i)),
                     filetype='cnt',
                     load_path=get_files_in_folder(dir_load_sub),
                     save_path=dir_save,
                     bad_channel_path=os.path.join(dir_load_sub, 'bads.txt')))
    return filepaths


class Test(LetterDelayMatch):
    def pipeline(self, filepaths):
        self.read_raw(filepaths)
        self.preprocess()
        self.make_epochs()
        self.make_data()
        self.save_epochs()
        self.save_data()
        self.save_metadata()


if __name__ == "__main__":
    dir_load = r'E:\Data\DelayMatch\letter'
    dir_save = r'E:\Data\tmp\ica_rest_250sfreq_baseline'
    group_size = 5
    sub_group = np.array_split(np.arange(1, 21), group_size)

    params = dict(tmin=0.,
                  tmax=5.5,
                  baseline=(-0.5, 0),
                  filter_low=1,
                  filter_high=None,
                  resample=250,
                  remove_eog=True)

    for subs in sub_group:
        filepaths = make_filepath(dir_load, dir_save, subs)
        srt = Test(**params)
        srt.pipeline(filepaths)
