"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_cryptography.py
Description: A module for cryptography
"""


import os
import argparse
import zlib
import pickle
from typing import Union, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
from vujade import vujade_bytes as bytes_
from vujade import vujade_path as path_
from vujade import vujade_str as str_
from vujade.vujade_debug import printd


class SymmetricCryptography(object):
    """
    Usage:
        param_spath_pem_public = './key_public.pem'
        key = crypto_.SymmetricCryptography.generate_pem(param_spath_pem_public)
        data_1 = 'Test Data 1'
        data_2 = np.array([1, 2, 3], dtype=np.int64)
        data = data_1  # or data_2

        data_encrypt = crypto_.SymmetricCryptography.encrypt(_spath_pem_public=param_spath_pem_public, _data=data, _is_to_bytes=True)
        data_decrypt = crypto_.SymmetricCryptography.decrypt(_spath_pem_public=param_spath_pem_public, _data=data_encrypt)

        if data == data_decrypt:
            print('The decryption is successful.')
        else:
            print('The decryption is failed.')
    """

    @classmethod
    def encrypt(cls, _spath_pem_public: str, _data: Any, _is_to_bytes: bool = True) -> dict:
        key = cls.__read_pem(_spath_pem=_spath_pem_public)
        if isinstance(key, bytes) is False:
            raise ValueError('The type of the _key should be bytes, not {}.'.format(type(key)))

        fernet = Fernet(key)

        if isinstance(_data, bytes) is False:
            _is_to_bytes = True

        if _is_to_bytes is True:
            data_bytes = pickle.dumps(_data)
        else:
            data_bytes = _data

        res = {'encrypt': fernet.encrypt(data_bytes), 'is_to_bytes': _is_to_bytes}

        return res

    @classmethod
    def decrypt(cls, _spath_pem_public: Optional[str], _data: dict, _key: Optional[bytes] = None) -> Any:
        if _spath_pem_public is None and _key is None:
            raise ValueError('One of the parameter between _spath_pem_public and _key should not be None.')

        if _key is None:
            key = cls.__read_pem(_spath_pem=_spath_pem_public)
        else:
            key = _key

        if isinstance(key, bytes) is False:
            raise ValueError('The type of the _key should be bytes, not {}.'.format(type(key)))

        fernet = Fernet(key)

        if (not 'encrypt' in _data.keys()) and (not 'is_to_bytes' in _data.keys()):
            raise ValueError('The _data may be incorrect.')

        data_bytes = fernet.decrypt(_data['encrypt'])

        if _data['is_to_bytes'] is True:
            res = pickle.loads(data_bytes)
        else:
            res = data_bytes

        return res

    @classmethod
    def __read_pem(cls, _spath_pem: str) -> bytes:
        path_pem = path_.Path(_spath=_spath_pem)
        if path_pem.path.is_file() is True:
            with open(path_pem.str, 'rb') as f:
                key = f.read()
        else:
            raise FileNotFoundError('The _spath_key, {} is not existed.'.format(_spath_pem))

        return key

    @classmethod
    def generate_pem(cls, _spath_pem_public: str) -> None:
        path_pem_public = path_.Path(_spath=_spath_pem_public)
        path_pem_public.unlink(_missing_ok=True)

        key_public = cls.__generate_key()

        with open(_spath_pem_public, mode='wb') as f:
            f.write(key_public)

    @classmethod
    def __generate_key(cls) -> bytes:
        return Fernet.generate_key()


class AsymmetricCryptography(object):
    """
    Usage:
        param_spath_pem_private_fake = './key_private_fake.pem'
        param_spath_pem_private = './key_private.pem'
        param_spath_pem_public = './key_public.pem'
        param_seed = 1024

        crypto_.AsymmetricCryptography.generate_pems(_spath_pem_private=param_spath_pem_private, _spath_pem_public=param_spath_pem_public, _seed=param_seed)
        data_1 = 'Test Data 1'
        data_2 = np.array([1, 2, 3], dtype=np.int64)
        data = data_1  # or data_2

        data_encrypt = crypto_.AsymmetricCryptography.encrypt(_spath_pem_public=param_spath_pem_public, _data=data, _is_to_bytes=True)
        data_decrypt = crypto_.AsymmetricCryptography.decrypt(_spath_pem_private=param_spath_pem_private, _data=data_encrypt)

        if data == data_decrypt:
            print('The decryption is successful.')
        else:
            print('The decryption is failed.')

        encrypt_key = str(AsymmetricEncryptDecrypt.encrypt(param_spath_pem_public, 'KEY_TO_BE_ENCRYPTED'.encode('utf-8')))
        decrypt_key = AsymmetricEncryptDecrypt.decrypt(param_spath_pem_private, _data_byte=ast.literal_eval(encrypt_key)).decode('utf-8')
    """

    @classmethod
    def encrypt(cls, _spath_pem_public: str, _data: Any, _is_to_bytes: bool = True) -> dict:
        key_public = cls.__read_pem(_spath_pem=_spath_pem_public)
        cipher = PKCS1_OAEP.new(key_public)

        if isinstance(_data, bytes) is False:
            _is_to_bytes = True

        if _is_to_bytes is True:
            data_bytes = pickle.dumps(_data)
        else:
            data_bytes = _data

        try:
            res = {'encrypt': cipher.encrypt(message=data_bytes), 'is_to_bytes': _is_to_bytes}
        except Exception as e:
            raise RuntimeError('It fails to encrypt the data. Exception: {}'.format(e))

        return res

    @classmethod
    def decrypt(cls, _spath_pem_private: str, _data: dict) -> Any:
        if (not 'encrypt' in _data.keys()) and (not 'is_to_bytes' in _data.keys()):
            raise ValueError('The _data may be incorrect.')

        key_private = cls.__read_pem(_spath_pem=_spath_pem_private)
        cipher = PKCS1_OAEP.new(key_private)
        data_bytes = cipher.decrypt(_data['encrypt'])

        if _data['is_to_bytes'] is True:
            res = pickle.loads(data_bytes)
        else:
            res = data_bytes

        return res

    @classmethod
    def generate_pems(cls, _spath_pem_private: str, _spath_pem_public: str, _seed: int = 1024) -> None:
        path_pem_private = path_.Path(_spath=_spath_pem_private)
        path_pem_public = path_.Path(_spath=_spath_pem_public)
        path_pem_private.unlink(_missing_ok=True)
        path_pem_public.unlink(_missing_ok=True)

        key_private, key_public = cls.__generate_keys(_seed=_seed)

        with open(_spath_pem_private, mode='wb') as f:
            f.write(key_private.exportKey('PEM'))

        with open(_spath_pem_public, mode='wb') as f:
            f.write(key_public.exportKey('PEM'))

    @classmethod
    def __read_pem(cls, _spath_pem: str) -> RSA.RsaKey:
        path_pem = path_.Path(_spath=_spath_pem)
        if path_pem.path.is_file() is True:
            with open(path_pem.str, 'rb') as f:
                key = RSA.importKey(f.read())
        else:
            raise FileNotFoundError('The _spath_key, {} is not existed.'.format(_spath_pem))

        return key

    @classmethod
    def __generate_keys(cls, _seed: int = 1024) -> tuple:
        key_private = RSA.generate(_seed)
        key_public = key_private.publickey()

        return key_private, key_public


class AsymmetricSignature(object):
    """
    Usage:
        data_encrypt_fake = {'encrypt': {'encrypt': b"\xa5\xde\x15\xfc\xc0BE\xb788", 'is_to_bytes': False}, 'is_to_bytes': True}
        param_spath_pem_private = './key_private.pem'
        param_spath_pem_public = './key_public.pem'
        param_seed = 1024

        crypto_.AsymmetricSignature.generate_pems(_spath_pem_private=param_spath_pem_private, _spath_pem_public=param_spath_pem_public, _seed=param_seed)
        data_1 = 'Test Data 1'
        data_2 = np.array([1, 2, 3], dtype=np.int64)
        data = data_1  # or data_2

        data_encrypt = crypto_.AsymmetricSignature.sign(_spath_pem_private=param_spath_pem_private, _data=data, _is_to_bytes=True)
        data_decrypt = crypto_.AsymmetricSignature.verify(_spath_pem_public=param_spath_pem_public, _data=data_encrypt)

        if data_decrypt['is_verified'] is True:
            print("The data_decrypt['is_verified']: {}.".format(data_decrypt['is_verified']))
            print("The data_decrypt['decrypt']:     {}.".format(data_decrypt['decrypt']))
        else:
            print("The data_decrypt['is_verified']: {}.".format(data_decrypt['is_verified']))
    """

    @classmethod
    def sign(cls, _spath_pem_private: str, _data: Any, _is_to_bytes: bool = True) -> dict:
        if isinstance(_data, bytes) is False:
            _is_to_bytes = True

        if _is_to_bytes is True:
            data_bytes = pickle.dumps(_data)
        else:
            data_bytes = _data

        signature = AsymmetricCryptography.encrypt(_spath_pem_public=_spath_pem_private, _data=data_bytes, _is_to_bytes=False)

        return {'encrypt': signature, 'is_to_bytes': _is_to_bytes}

    @classmethod
    def verify(cls, _spath_pem_public: str, _data: dict) -> dict:
        if (not 'encrypt' in _data.keys()) and (not 'is_to_bytes' in _data.keys()):
            raise ValueError('The _data may be incorrect.')

        try:
            # Verified
            data_bytes = AsymmetricCryptography.decrypt(_spath_pem_private=_spath_pem_public, _data=_data['encrypt'])
            if _data['is_to_bytes'] is True:
                data = pickle.loads(data_bytes)
            else:
                data = data_bytes
            is_verified = True
        except Exception as e:
            # Counterfeited
            data = None
            is_verified = False

        return {'is_verified': is_verified, 'decrypt': data}

    @classmethod
    def generate_pems(cls, _spath_pem_private: str, _spath_pem_public: str, _seed: int = 1024) -> None:
        path_pem_private = path_.Path(_spath=_spath_pem_private)
        path_pem_public = path_.Path(_spath=_spath_pem_public)
        path_pem_private.unlink(_missing_ok=True)
        path_pem_public.unlink(_missing_ok=True)

        key_private, key_public = cls.__generate_keys(_seed=_seed)

        with open(_spath_pem_public, mode='wb') as f:
            f.write(key_private.exportKey('PEM'))

        with open(_spath_pem_private, mode='wb') as f:
            f.write(key_public.exportKey('PEM'))

    @classmethod
    def __read_pem(cls, _spath_pem: str) -> RSA.RsaKey:
        path_pem = path_.Path(_spath=_spath_pem)
        if path_pem.path.is_file() is True:
            with open(path_pem.str, 'rb') as f:
                key = RSA.importKey(f.read())
        else:
            raise FileNotFoundError('The _spath_key, {} is not existed.'.format(_spath_pem))

        return key

    @classmethod
    def __generate_keys(cls, _seed: int = 1024) -> tuple:
        key_private = RSA.generate(_seed)
        key_public = key_private.publickey()

        return key_private, key_public


class EncryptDecryptFiles(object):
    def __init__(self, _spath_root_encrypt: str = './encrypt', _spath_root_decrypt: str = './decrypt', _spath_root_key: str = './key', _is_to_bytes: bool = False) -> None:
        super(EncryptDecryptFiles, self).__init__()
        self.path_root_encrypt = path_.Path(_spath_root_encrypt)
        self.path_root_decrypt = path_.Path(_spath_root_decrypt)
        self.path_root_key = path_.Path(_spath_root_key)
        self.is_to_bytes = _is_to_bytes
        self.path_pem_private = path_.Path('{}/key_private.pem'.format(self.path_root_key.str))
        self.path_pem_public = path_.Path('{}/key_public.pem'.format(self.path_root_key.str))
        self.name_key_symmetric = 'key_symmetric.pem'

        if (self.is_to_bytes is True):
            raise NotImplementedError('The parameter, _is_to_bytes should be True.')

    def encrypt(self, _spath_root_to_be_encrypted: str):
        self.path_root_encrypt.rmtree()
        self.path_root_encrypt.path.mkdir( mode=0o777, parents=True, exist_ok=True)

        path_root_to_be_encrypted = path_.Path(_spath_root_to_be_encrypted)
        spath_key_symmetric = '{}/{}'.format(path_root_to_be_encrypted.str, self.name_key_symmetric)
        SymmetricCryptography.generate_pem(spath_key_symmetric)

        for _idx, _path_file_to_be_encrypted in enumerate(path_root_to_be_encrypted.path.glob('*')):
            path_file_to_be_encrypted = path_.Path(str(_path_file_to_be_encrypted))
            if path_file_to_be_encrypted.path.is_dir() is True:
                raise NotImplementedError('The sub-directory, {} has not been supported yet.')

            if path_file_to_be_encrypted.filename != self.name_key_symmetric:
                filename_encrypted = SymmetricCryptography.encrypt(_spath_pem_public=spath_key_symmetric, _data=path_file_to_be_encrypted.filename.encode('utf-8'), _is_to_bytes=self.is_to_bytes)
                path_file_encrypted = path_.Path('{}/{}'.format(self.path_root_encrypt.str, filename_encrypted['encrypt'].decode('utf-8')))
                data_original = self.__read_file(_spath_file=path_file_to_be_encrypted.str)
                data_encrypted = SymmetricCryptography.encrypt(_spath_pem_public=spath_key_symmetric, _data=data_original, _is_to_bytes=self.is_to_bytes)
            else:
                filename_encrypted = 'enc_{}'.format(self.name_key_symmetric)
                path_file_encrypted = path_.Path('{}/{}'.format(self.path_root_encrypt.str, filename_encrypted))
                data_original = self.__read_file(_spath_file=path_file_to_be_encrypted.str)
                data_encrypted = AsymmetricCryptography.encrypt(_spath_pem_public=self.path_pem_public.str, _data=data_original, _is_to_bytes=self.is_to_bytes)

            self.__write_file(_spath_file=path_file_encrypted.str, _data_encrypt_content=data_encrypted['encrypt'])

    def decrypt(self):
        self.path_root_decrypt.rmtree()
        self.path_root_decrypt.path.mkdir( mode=0o777, parents=True, exist_ok=True)

        filename_enc_key_symmetric = 'enc_{}'.format(self.name_key_symmetric)
        path_enc_key_symmetric = path_.Path('{}/{}'.format(self.path_root_encrypt.str, filename_enc_key_symmetric))

        if path_enc_key_symmetric.path.is_file() is False:
            raise FileNotFoundError

        enc_key_symmetric = self.__read_file(_spath_file=path_enc_key_symmetric.str)
        key_symmetric = AsymmetricCryptography.decrypt(_spath_pem_private=self.path_pem_private.str, _data=self.__make_data_for_symmetriccryptography(enc_key_symmetric))

        for _idx, _path_file_encrypted in enumerate(self.path_root_encrypt.path.glob('*')):
            path_file_encrypted = path_.Path(str(_path_file_encrypted))
            if path_file_encrypted.filename == 'enc_{}'.format(self.name_key_symmetric):
                continue

            filename_encrypted = path_file_encrypted.filename
            data_encrypted = self.__read_file(_spath_file=path_file_encrypted.str)
            filename_decrypted = SymmetricCryptography.decrypt(_spath_pem_public=None, _data=self.__make_data_for_symmetriccryptography(filename_encrypted), _key=key_symmetric)
            data_decrypted = SymmetricCryptography.decrypt(_spath_pem_public=None, _data=self.__make_data_for_symmetriccryptography(data_encrypted), _key=key_symmetric)

            path_file_to_be_written = path_.Path('{}/{}'.format(self.path_root_decrypt.str, filename_decrypted.decode('utf-8')))
            self.__write_file(_spath_file=path_file_to_be_written.str, _data_encrypt_content=data_decrypted)

    def generate_pems(self) -> None:
        AsymmetricCryptography.generate_pems(
            _spath_pem_private=self.path_pem_private.str,
            _spath_pem_public=self.path_pem_public.str,
            _seed=1024
        )

    def __read_file(self, _spath_file: str) -> bytes:
        with open(_spath_file, 'rb') as f:
            res = f.read()
        return res

    def __write_file(self, _spath_file: str, _data_encrypt_content: bytes) -> None:
        with open(_spath_file, 'wb') as f:
            f.write(_data_encrypt_content)

    def __make_data_for_symmetriccryptography(self, _data_encrypt_content: bytes) -> dict:
        return {'encrypt': _data_encrypt_content, 'is_to_bytes': self.is_to_bytes}


class SymmetricCryptographyBytesBytearray(object):
    magic_bytes = b'aa1b2848-c32f-4c34-b80c-cfcdd39b56a3'

    @classmethod
    def encrypt(cls, _bytes_bytearr: Union[bytes, bytearray], _key: bytes) -> bytes:
        # Encrypt the bytes or bytearray using AES-CBC with PKCS7 padding
        cipher = Cipher(algorithms.AES(_key), modes.CBC(cls.magic_bytes[:16]), backend=default_backend())
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_bytes = padder.update(_bytes_bytearr) + padder.finalize()
        encrypted_bytes = encryptor.update(padded_bytes) + encryptor.finalize()

        return encrypted_bytes

    @classmethod
    def decrypt(cls, _bytes_enc: bytes, _key: bytes) -> bytes:
        # Decrypt the encrypted bytes using AES-CBC with PKCS7 padding
        cipher = Cipher(algorithms.AES(_key), modes.CBC(cls.magic_bytes[:16]), backend=default_backend())
        decryptor = cipher.decryptor()
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        decrypted_padded_bytes = decryptor.update(_bytes_enc) + decryptor.finalize()
        decrypted_bytearr = unpadder.update(decrypted_padded_bytes) + unpadder.finalize()

        return decrypted_bytearr

    @classmethod
    def generate_key(cls) -> bytes:
        # Generate a random 256-bit encryption and decryption key
        return os.urandom(32)


class MainSymmetricCryptographyBytesBytearray(object):
    @classmethod
    def run(cls) -> None:
        args = cls._get_args()

        path_file_src = path_.Path(args.path_file_src)

        if args.mode == 'enc':
            path_file_dst = path_file_src.replace_ext('.scb')
        elif args.mode == 'dec':
            if path_file_src.ext == '.scb':
                if args.path_file_dst is None:
                    raise ValueError('The path_file_dst should be assigned in the dec mode.')
                else:
                    path_file_dst =path_.Path(args.path_file_dst)
            else:
                raise ValueError("The extension of the path_file_src should be '.scb' in the dec mode.")
        else:
            raise ValueError('The given args.mode has not been supported.')

        if args.mode == 'enc':
            bytes_ori = bytes_.Bytes.read(_spath_file=path_file_src.str)
            bytes_enc = SymmetricCryptographyBytesBytearray.encrypt(_bytes_bytearr=bytes_ori, _key=args.key)
            if args.is_compress is True:
                bytes_enc = zlib.compress(bytes_enc)
            bytes_.Bytes.write(_spath_file=path_file_dst.str, _bytes=bytes_enc)
        else:
            bytes_enc = bytes_.Bytes.read(_spath_file=path_file_src.str)
            if args.is_compress is True:
                bytes_enc = zlib.decompress(bytes_enc)
            bytes_dec = SymmetricCryptographyBytesBytearray.decrypt(_bytes_enc=bytes_enc, _key=args.key)
            bytes_.Bytes.write(_spath_file=path_file_dst.str, _bytes=bytes_dec)

    @staticmethod
    def _get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Main function for calling SymmetricCryptographyBytesBytearray.')
        parser.add_argument('--mode', type=str, default='enc', help='i) enc; ii) dec')
        parser.add_argument('--path_file_src', type=str, default='./_bks/data/img_test.png')
        parser.add_argument('--path_file_dst', type=str, default=None)
        parser.add_argument('--is_compress', type=str_.str2bool, default=True)
        parser.add_argument('--key', type=bytes, required=True, help='SymmetricCryptographyBytesBytearray.generate_key()')
        args = parser.parse_args()

        return args
