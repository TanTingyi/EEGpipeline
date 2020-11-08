"""
数据集的基类
Authors: Matthew Tan <5636374@qq.com>
Time: 2020.11.7
"""
import abc


class BaseDataset(metaclass=abc.ABCMeta):
    """BaseDataset
    """
    def __init__(self, code):
        """
        Parameters
        ----------
        code: string
            数据集的唯一标号
        """
        self.code = code

    def _run_ica(self, raw):
        """ICA 去伪迹
        Parameters         
        ----------    
        raw : mne.io.Raw
            原始数据
        Returns         
        -------
        raw : mne.io.Raw
            去除了伪迹的数据
        """
        pass

    def _run_reference(self, raw):
        """重参考
        Parameters         
        ----------    
        raw : mne.io.Raw
            原始数据
        Returns         
        -------
        raw : mne.io.Raw
            重参考后的数据
        """
        pass

    def _run_channel_repair_exclud(self, raw):
        """坏道修复和通道去除
        Parameters         
        ----------    
        raw : mne.io.Raw
            原始数据
        Returns         
        -------
        raw : mne.io.Raw
            修复坏道后的数据
        """
        pass

    @abc.abstractmethod
    def get_raw(self, paths, bad_path):
        """加载 cnt 文件
        Parameters         
        ----------
        paths : str | list            
            cnt 文件的路径，其可以是一个 list，这代表着
            将所有的 cnt 拼接为一个文件，使用相同的
            预处理参数，建议将同一个人的数据统一处理
            paths = ['s1_1_1.cnt',
                    's1_2_1.cnt',
                    's1_3_1.cnt']
        bad_path : None | str 
            坏道记录文件的路径，坏道描述以空格隔开
        Returns         
        -------
        raw : mne.io.Raw
            未经预处理的原始数据
        """
        pass

    @abc.abstractmethod
    def preprocess(self, raws):
        """推荐的预处理流程：
            坏道插值 -> 带通滤波 -> ICA 去眼电 -> 重参考
        Parameters         
        ----------    
        raws : mne.io.Raw | list
            当输入为单个被试的原始数据时，会单独寻找被试的眼电
            成分，可能会浪费大量的时间用于检查每个单独被试的 IC。
            如果输入为 list，就可以使用模板匹配的方法去除眼电，
            仅需要找到一个眼电 IC 的模板。
            但是模板匹配会占据大量内存，可以每次使用 3 - 5
            个被试的数据，避免爆内存导致的 CPU 计算效率低。
        Returns         
        -------
        raws : mne.io.Raw | list
            预处理后的数据
        """
        pass

    @abc.abstractmethod
    def get_epochs(self, raw):
        """提取 trail
        Parameters:
        ----------
        raw : mne.io.Raw
            经过预处理后的数据
        Return:
        -------
        epochs : mne.Epochs
            数据进一步预处理后，分 trail 
        """
        pass

    def get_data(self, epochs):
        """获取预处理好的数据
        Parameters:
        ----------
        epochs : mne.Epochs
            分好 trail 的数据
        Return:
        -------
        data : np.array, shape (trail, channels, nsamples)
            预处理后提取的数据         
        label : np.array, shape (trail)             
            数据对应的标签         
        """
        pass

    def pipeline(self, *args, **kwargs):
        """数据处理 pipeline，定义数据的处理流程
        """
        pass
    
    def _event_tranform(self, events_old, event_id_old):
        """所以为了使得标签易于理解，需要对标签进行转换。
        Parameters         
        ----------
        events_old : array, shape (n_events, 3)
            mne.events_from_annotations 得到的 events
        event_id_old : dict
            mne.events_from_annotations 得到的 events_id
        Returns
        -------
        events_new : array, shape (n_events, 3)
            转换标签后的 events
        event_id_new : dict
            转换标签后的 events_id
        """
        return events_old, event_id_old
