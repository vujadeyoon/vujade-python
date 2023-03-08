import os
import argparse
from tqdm import tqdm
from vujade import vujade_bytes as bytes_
from vujade import vujade_path as path_
from vujade import vujade_random as rand_
from vujade import vujade_str as str_
from vujade.vujade_debug import printd


class MainEraseData(object):
    @classmethod
    def run(cls) -> None:
        args = cls._get_args()

        path_src = path_.Path(args.path_src)

        if path_src.path.is_file() is True:
            path_rglob = (path_src.path, )
        elif path_src.path.is_dir() is True:
            path_rglob = tuple(path_src.path.rglob(pattern='*'))
        else:
            raise ValueError('The given path_src be file or directory path.')

        for _idx, _path_element in enumerate(tqdm(path_rglob)):
            path_element = path_.Path(str(_path_element))
            if path_element.path.is_file() is True:
                cls._erase_file(
                    _spath_file=path_element.str,
                    _num_iter=args.num_iter,
                    _is_rename=args.is_rename,
                    _magic_number=args.magic_number
                )
            elif path_element.path.is_dir() is True:
                continue
            else:
                ValueError('The path_element be file or directory path.')

        if (path_src.path.is_dir() is True) and (args.is_rename is True):
            path_src = path_.Path(args.path_src)
            path_rglob = sorted((tuple(path_src.path.rglob(pattern='*'))))[::-1]

            for _idx, _path_element in enumerate(tqdm(path_rglob)):
                path_element = path_.Path(str(_path_element))
                if path_element.path.is_dir() is True:
                    path_element_new = path_.Path(os.path.join(path_element.parent.str, rand_.get_random_string(_num_len_str=args.magic_number)))
                    path_element.move(_spath_dst=path_element_new.str)

    @classmethod
    def _erase_file(cls, _spath_file: str, _num_iter: int = 1, _is_rename: bool = True, _magic_number: int = 10) -> None:
        path_file = path_.Path(_spath_file)

        len_bytes = len(bytes_.Bytes.read(_spath_file=_spath_file))
        for _ in range(_num_iter):
            bytes_.Bytes.write(_spath_file=_spath_file, _bytes=os.urandom(len_bytes))
        bytes_.Bytes.write(_spath_file=_spath_file, _bytes=os.urandom(_magic_number))

        if _is_rename is True:
            path_file_new = path_file.replace(_old=path_file.filename, _new=cls._get_filename_new(_magic_number=_magic_number))
            path_file.move(_spath_dst=path_file_new.str)

    @staticmethod
    def _get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--path_src', type=str, required=True)
        parser.add_argument('--num_iter', type=int, default=10)
        parser.add_argument('--is_rename', type=str_.str2bool, default=True, help='The is_rename is used to hide the file name and extension for a target file.')
        parser.add_argument('--magic_number', type=int, default=10, help='The magic_number is used to hide the length of bytes and file name for a target file.')
        args = parser.parse_args()

        return args

    @staticmethod
    def _get_filename_new(_magic_number: int = 10) -> str:
        return '{}.{}'.format(rand_.get_random_string(_num_len_str=_magic_number), rand_.get_random_string(_num_len_str=3))


if __name__=='__main__':
    # Usage:
    #     i)   export PYTHONPATH=$PYTHONPATH:$(pwd)
    #     ii)  python3 ./vujade/vujade_erase.py --path_src ${path_dir_or_file}
    #     iii) wipe -rfi ${path_dir_or_file}

    MainEraseData.run()
