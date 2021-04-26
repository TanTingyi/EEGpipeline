from copy import deepcopy
from os.path import isfile
from pathlib import Path


def _depth_count(x):
    """Return maximum depth of the returned list.

    Parameters
    ----------
    x : list.

    Return
    ------
    ans : int
        maximum depth.
    """
    return int(isinstance(x,
                          list)) and len(x) and 1 + max(map(_depth_count, x))


def check_paths(paths):
    """Return path list make sure load_path is a list and all paths exit.

    Parameters
    ----------
    paths : FilePath | list of FilePath.
        FilePath object to be detected.

    Return
    ------
    paths_checked : list.
        List of FilePaths that conform to the specification
    """

    if _depth_count(paths) < 1:
        paths_checked = [paths]
    else:
        paths_checked = deepcopy(paths)
    for path in paths_checked:
        if not isinstance(path.load_path, list):
            path.load_path = [path.load_path]

        file_uncheck = path.load_path.copy()
        if path.bad_channel_path:
            file_uncheck.append(path.bad_channel_path)
        _files_not_exist = _find_path_not_exist(file_uncheck)

        if _files_not_exist:
            raise RuntimeError('These files do not exit:', *_files_not_exist)
    return paths_checked


def _find_path_not_exist(paths):
    """Recursively check whether the path in the list exists.

    takes a list as input and return path if it is not exist.

    Parameters
    ----------
    paths : list.
        list containing paths.

    Return
    ------
    paths_not_exist : list
        paths which not exist.
    """
    paths_not_exist = []

    def _check_file(paths):
        if not isinstance(paths, list):
            if not isfile(paths):
                return paths
        else:
            for path in paths:
                ans = _check_file(path)
                if isinstance(ans, str):
                    paths_not_exist.append(ans)

    _check_file(paths)
    return paths_not_exist


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
