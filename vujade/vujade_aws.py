"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_aws.py
Description: A module for the Amazon Web Services (AWS)
"""


import boto3
from vujade import vujade_utils as utils_


def set_credential(_access_key: str, _secret_key: str, _path_aws: str = '/root/.aws') -> bool:
    res = True

    utils_.makedirs(_path=_path_aws, _exist_ok=True)
    res_1 = utils_.run_command('echo \'[default]\' >> {}/credentials'.format(_path_aws))
    res_2 = utils_.run_command('echo \'aws_access_key_id = ' + _access_key + '\' >> {}/credentials'.format(_path_aws))
    res_3 = utils_.run_command('echo \'aws_secret_access_key = ' + _secret_key + '\' >> {}/credentials'.format(_path_aws))

    if (res_1['is_success'] == False) or (res_2['is_success'] == False) or (res_3['is_success'] == False):
        res = False

    return res


def get_s3_client(_access_key: str, _secret_key: str):
    try:
        s3 = boto3.client('s3', aws_access_key_id=_access_key, aws_secret_access_key=_secret_key)
    except Exception as e:
        raise ConnectionError('The S3 connection is failed with the exception: {}.'.format(e))

    return s3


def upload_directory(_s3, _bucket_name: str, _remote_name: str, _local_name: str , _ext_file='.png') -> bool:
    try:
        list_files_local = utils_.get_glob(_path=_local_name, _ext_file=_ext_file)
        for idx, file_local in enumerate(list_files_local):
            file_remote = file_local.replace(_local_name, _remote_name)
            _s3.upload_file(file_local, _bucket_name, file_remote)
        res = True
    except Exception as e:
        print('The uploading for the S3 is failed with the excpetion: {}.'.format(e))
        res = False

    return res


def upload_file(_s3, _bucket_name, _remote_name, _local_name):
    try:
        _s3.upload_file(_local_name, _bucket_name, _remote_name)
        res = True
    except Exception as e:
        print('The uploading for the S3 is failed with the excpetion: {}.'.format(e))
        res = False

    return res


def download_file(_s3, _bucket_name, _remote_name, _local_name):
    try:
        res = utils_.rmfile(_path_file=_local_name)
        if res == 1:
            print('The removing a file, {}, is succeeded.'.format(_local_name))
        elif res == -1:
            print('The removing a file, {}, is failed.'.format(_local_name))
        else: # res == 0
            print('The file, {}, is not existed.'.format(_local_name))
        _s3.download_file(_bucket_name, _remote_name, _local_name)
        res = True
    except Exception as e:
        print('The downloading a file, {}, is failed with the excpetion: {}.'.format(_remote_name, e))
        res = False

    return res
