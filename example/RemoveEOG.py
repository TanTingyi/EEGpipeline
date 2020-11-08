import os

from smd.delay import Letter


class RemoveEOG(Letter):
    def pipeline(self, folder, save_path):
        """使用模板匹配的方法去除眼电，并保存结果
        """
        sub_folders = [
            os.path.join(folder, 's' + str(i)) for i in range(1, 21)
        ]
        raws = []
        n_sub = 4

        for i, sub_folder in enumerate(sub_folders):
            sub = i + 1
            file_paths = self._get_files_in_folder(sub_folder)
            raw = self.get_raw(file_paths,
                               os.path.join(sub_folder, 'bads.txt'))
            raws.append(raw)
            if len(raws) % n_sub == 0:
                raws = self.preprocess(raws)
                for j, raw in enumerate(raws):
                    raw.save(
                        os.path.join(save_path,
                                     str(sub + 1 - (n_sub - j)) + '_raw.fif'))
                raws = []


if __name__ == "__main__":
    folder = r'E:\Data\DelayMatch\letter'
    save_path = r'E:\Data\DelayMatch\letter\data_ica_rest'

    dm = RemoveEOG(tmin=0.,
                   tmax=3.3,
                   baseline=(-0.2, 0),
                   reject=dict(eeg=200e-6),
                   filter_low=0.5,
                   filter_high=45.,
                   resample=160,
                   verbose=0)
    dm.pipeline(folder, save_path)
