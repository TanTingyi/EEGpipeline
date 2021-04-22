from collections import namedtuple


class FilePath(
        namedtuple('FilePath',
                   ['subject', 'filetype', 'load_path', 'save_path', 'bad_channel_path'])):
    __slots__ = ()

    def __new__(cls, subject, filetype, load_path, save_path, bad_channel_path=None):
        return super(FilePath, cls).__new__(cls, subject, filetype, load_path, save_path,
                                            bad_channel_path)
