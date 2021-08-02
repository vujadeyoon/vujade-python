"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_cryptography.py
Description: A module for cryptography
"""


import pickle
from typing import Any
from cryptography.fernet import Fernet
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
from vujade import vujade_path as path_


class SymmetricCryptography(object):
    """
    Usage:
        key = crypto_.SymmetricCryptography.generate_key()
        data_1 = 'Test Data 1'
        data_2 = np.array([1, 2, 3], dtype=np.int64)
        data = data_1  # or data_2

        data_encrypt = crypto_.SymmetricCryptography.encrypt(_key=key, _data=data, _to_bytes=True)
        data_decrypt = crypto_.SymmetricCryptography.decrypt(_key=key, _data=data_encrypt)

        if data == data_decrypt:
            print('The decryption is successful.')
        else:
            print('The decryption is failed.')
    """

    @classmethod
    def encrypt(cls, _key: bytes, _data: Any, _to_bytes: bool = True) -> dict:
        if isinstance(_key, bytes) is False:
            raise ValueError('The type of the _key should be bytes, not {}.'.format(type(_key)))

        fernet = Fernet(_key)

        if isinstance(_data, bytes) is False:
            _to_bytes = True

        if _to_bytes is True:
            data_bytes = pickle.dumps(_data)
        else:
            data_bytes = _data

        res = {'encrypt': fernet.encrypt(data_bytes), 'to_bytes': _to_bytes}

        return res

    @classmethod
    def decrypt(cls, _key: bytes, _data: dict) -> Any:
        if isinstance(_key, bytes) is False:
            raise ValueError('The type of the _key should be bytes, not {}.'.format(type(_key)))

        fernet = Fernet(_key)

        if (not 'encrypt' in _data.keys()) and (not 'to_bytes' in _data.keys()):
            raise ValueError('The _data may be incorrect.')

        data_bytes = fernet.decrypt(_data['encrypt'])

        if _data['to_bytes'] is True:
            res = pickle.loads(data_bytes)
        else:
            res = data_bytes

        return res

    @classmethod
    def generate_key(cls) -> bytes:
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

        data_encrypt = crypto_.AsymmetricCryptography.encrypt(_spath_pem_public=param_spath_pem_public, _data=data, _to_bytes=True)
        data_decrypt = crypto_.AsymmetricCryptography.decrypt(_spath_pem_private=param_spath_pem_private, _data=data_encrypt)

        if data == data_decrypt:
            print('The decryption is successful.')
        else:
            print('The decryption is failed.')
    """

    @classmethod
    def encrypt(cls, _spath_pem_public: str, _data: Any, _to_bytes: bool = True) -> dict:
        key_public = cls.__read_pem(_spath_pem=_spath_pem_public)
        cipher = PKCS1_OAEP.new(key_public)

        if isinstance(_data, bytes) is False:
            _to_bytes = True

        if _to_bytes is True:
            data_bytes = pickle.dumps(_data)
        else:
            data_bytes = _data

        try:
            res = {'encrypt': cipher.encrypt(message=data_bytes), 'to_bytes': _to_bytes}
        except Exception as e:
            raise RuntimeError('It fails to encrypt the data. Exception: {}'.format(e))

        return res

    @classmethod
    def decrypt(cls, _spath_pem_private: str, _data: dict) -> Any:
        if (not 'encrypt' in _data.keys()) and (not 'to_bytes' in _data.keys()):
            raise ValueError('The _data may be incorrect.')

        key_private = cls.__read_pem(_spath_pem=_spath_pem_private)
        cipher = PKCS1_OAEP.new(key_private)
        data_bytes = cipher.decrypt(_data['encrypt'])

        if _data['to_bytes'] is True:
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
    def __read_pem(cls, _spath_pem: str) -> RSA._RSAobj:
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
        data_encrypt_fake = {'encrypt': {'encrypt': b"\xa5\xde\x15\xfc\xc0BE\xb788", 'to_bytes': False}, 'to_bytes': True}
        param_spath_pem_private = './key_private.pem'
        param_spath_pem_public = './key_public.pem'
        param_seed = 1024

        crypto_.AsymmetricSignature.generate_pems(_spath_pem_private=param_spath_pem_private, _spath_pem_public=param_spath_pem_public, _seed=param_seed)
        data_1 = 'Test Data 1'
        data_2 = np.array([1, 2, 3], dtype=np.int64)
        data = data_1  # or data_2

        data_encrypt = crypto_.AsymmetricSignature.sign(_spath_pem_private=param_spath_pem_private, _data=data, _to_bytes=True)
        data_decrypt = crypto_.AsymmetricSignature.verify(_spath_pem_public=param_spath_pem_public, _data=data_encrypt)

        if data_decrypt['is_verified'] is True:
            print("The data_decrypt['is_verified']: {}.".format(data_decrypt['is_verified']))
            print("The data_decrypt['decrypt']:     {}.".format(data_decrypt['decrypt']))
        else:
            print("The data_decrypt['is_verified']: {}.".format(data_decrypt['is_verified']))
    """

    @classmethod
    def sign(cls, _spath_pem_private: str, _data: Any, _to_bytes: bool = True) -> dict:
        if isinstance(_data, bytes) is False:
            _to_bytes = True

        if _to_bytes is True:
            data_bytes = pickle.dumps(_data)
        else:
            data_bytes = _data

        signature = AsymmetricCryptography.encrypt(_spath_pem_public=_spath_pem_private, _data=data_bytes, _to_bytes=False)

        return {'encrypt': signature, 'to_bytes': _to_bytes}

    @classmethod
    def verify(cls, _spath_pem_public: str, _data: dict) -> dict:
        if (not 'encrypt' in _data.keys()) and (not 'to_bytes' in _data.keys()):
            raise ValueError('The _data may be incorrect.')

        try:
            # Verified
            data_bytes = AsymmetricCryptography.decrypt(_spath_pem_private=_spath_pem_public, _data=_data['encrypt'])
            if _data['to_bytes'] is True:
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
    def __read_pem(cls, _spath_pem: str) -> RSA._RSAobj:
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
