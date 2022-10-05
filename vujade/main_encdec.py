"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: main_encdec.py
Description: A main python script to encrypt or decrypt files in a directory.
"""


import argparse
from vujade import vujade_cryptography as crypto_
from vujade import vujade_str as str_


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Encrypt and decrypt files.')
    parser.add_argument("--is_encrypt", action="store_true")
    parser.add_argument("--is_decrypt", action="store_true")
    parser.add_argument('--path_root_to_be_encrypted', type=str, default='./to_be_encrypted')
    parser.add_argument('--path_root_encrypt', type=str, default='./encrypt')
    parser.add_argument('--path_root_decrypt', type=str, default='./decrypt')
    parser.add_argument('--path_root_key', type=str, default='./key')
    parser.add_argument('--is_to_bytes', type=str_.str2bool, default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    is_encrypt = args.is_encrypt
    is_decrypt = args.is_decrypt

    if is_encrypt ^ is_decrypt is False:
        raise ValueError('You should select a mode: i) --is_encrypt; ii) --is_decrypt.')

    edf = crypto_.EncryptDecryptFiles(
        _spath_root_encrypt=args.path_root_encrypt,
        _spath_root_decrypt=args.path_root_decrypt,
        _spath_root_key=args.path_root_key,
        _is_to_bytes=args.is_to_bytes
    )

    if is_encrypt is True:
        edf.generate_pems()
        edf.encrypt(_spath_root_to_be_encrypted=args.path_root_to_be_encrypted)

    if is_decrypt is True:
        edf.decrypt()
