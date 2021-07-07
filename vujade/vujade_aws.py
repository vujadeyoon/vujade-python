"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_aws.py
Description: A module for the Amazon Web Services (AWS)
"""


import os
import sys
import argparse
import boto3
import botocore.client
from typing import Set, Optional
from pathlib import Path
try:
    from vujade import vujade_utils as utils_
    from vujade import vujade_path as path_
    from vujade import vujade_time as time_
    from vujade import vujade_text as text_
    from vujade import vujade_download as download_
    from vujade import vujade_compression as comp_
    from vujade.vujade_debug import printf
except Exception as e:
    sys.path.append(os.path.join(os.getcwd()))
    from vujade import vujade_utils as utils_
    from vujade import vujade_path as path_
    from vujade import vujade_time as time_
    from vujade import vujade_text as text_
    from vujade import vujade_download as download_
    from vujade import vujade_compression as comp_
    from vujade.vujade_debug import printf


def get_aws_mode_base() -> Set[str]:
    return {'awscli', 'enroll'}


def get_aws_mode_s3() -> Set[str]:
    return get_aws_mode_base() | {'upload', 'download', 'delete', 'all'}


def get_aws_mode_stepfunctions() -> Set[str]:
    return get_aws_mode_base() | {'all'}


class BaseAWS(object):
    def __init__(self, _spath_aws: str, _access_key: Optional[str] = None, _secret_key: Optional[str] = None):
        super(BaseAWS, self).__init__()
        self.path_aws = path_.Path(_spath=_spath_aws)
        if (_access_key is None) or (_secret_key is None):
            self.access_key, self.secret_key = self._get_aws_access_secret_keys()
        else:
            self.access_key, self.secret_key = _access_key, _secret_key

    def _get_client(self, _name_client: str) -> botocore.client:
        try:
            client = boto3.client(_name_client, aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key)
        except Exception as e:
            raise ConnectionError('It is failed to get a {} client with the exception: {}.'.format(_name_client, e))

        return client

    @staticmethod
    def install_awscli():
        url_awscliv2_zip = 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip'
        path_awscliv2_zip = path_.Path(_spath=os.path.join(os.getcwd(), 'awscliv2.zip'))
        path_awscliv2 = path_.Path(_spath=os.path.join(os.getcwd(), 'aws'))

        # Install the awscli.
        download_.Download.run(_url=url_awscliv2_zip, _spath_filename=path_awscliv2_zip.str)
        zip_awscliv2 = comp_.Zip()
        zip_awscliv2.decompress(_spath_zip=path_awscliv2_zip.str, _d=None)
        utils_.SystemCommand.run(_command='sudo {}/install'.format(path_awscliv2))

        # Remove the temporary file and directory.
        path_awscliv2_zip.unlink(_missing_ok=True)
        path_awscliv2.rmtree(_ignore_errors=False, _onerror=None)

    def _get_aws_access_secret_keys(self) -> tuple:
        path_credentials = path_.Path(_spath=os.path.join(self.path_aws.str, 'credentials'))

        if path_credentials.path.is_file() is True:
            res = dict()
            lines = text_.TEXT(_spath_filename=path_credentials.str, _mode='r').read_lines()
            for _idx, _line in enumerate(lines):
                _line = _line.rstrip('\n')
                if '=' in _line:
                    key, value = _line.split(' = ')
                    res[key] = value
        else:
            raise FileNotFoundError('The AWS CLI should be installed using the AWS.install_awscli().')

        return res['aws_access_key_id'], res['aws_secret_access_key']

    @utils_.deprecated
    def enroll_credentials(self, _spath_aws: str, _region: str = 'ap-northeast-2', _output: str = 'json') -> None:
        utils_.SystemCommand.run(_command='rm -rf {}'.format(_spath_aws))
        utils_.SystemCommand.run(_command='mkdir -p {}'.format(_spath_aws))
        utils_.SystemCommand.run(_command='echo \'[default]\' >> {}/credentials'.format(_spath_aws))
        utils_.SystemCommand.run(_command='echo \'aws_access_key_id = ' + self.access_key + '\' >> {}/credentials'.format(_spath_aws))
        utils_.SystemCommand.run(_command='echo \'aws_secret_access_key = ' + self.secret_key + '\' >> {}/credentials'.format(_spath_aws))
        utils_.SystemCommand.run(_command='echo \'[default]\' >> {}/config'.format(_spath_aws))
        utils_.SystemCommand.run(_command='echo \'region = ' + _region + '\' >> {}/config'.format(_spath_aws))
        utils_.SystemCommand.run(_command='echo \'output = ' + _output + '\' >> {}/config'.format(_spath_aws))


class S3(BaseAWS):
    def __init__(self, _spath_aws: str, _mode: str, _access_key: Optional[str] = None, _secret_key: Optional[str] = None):
        super(S3, self).__init__(_spath_aws=_spath_aws, _access_key=_access_key, _secret_key=_secret_key)
        self.mode = _mode

        if self.mode not in get_aws_mode_s3():
            raise NotImplementedError('The AWS S3 mode: {} is not supported yet.'.format(self.mode))

        if _mode in get_aws_mode_base():
            self.client = None
        else:
            self.client = self._get_client(_name_client='s3')

    def get_object(self, _name_bucket: str, _spath_remote: str) -> bool:
        self._check_spath_remote(_spath_remote=_spath_remote)

        try:
            self.client.get_object(Bucket=_name_bucket, Key=_spath_remote)
            res = True
        except Exception as e:
            print('The path, {} is not existed with the exception: {}.'.format(_spath_remote, e))
            res = False

        return res

    def delete_object(self, _name_bucket: str, _spath_remote: str) -> bool:
        self._check_spath_remote(_spath_remote=_spath_remote)

        try:
            self.client.delete_object(Bucket=_name_bucket, Key=_spath_remote)
            res = True
        except Exception as e:
            print('Deleting the object {} is failed with the exception: {}.'.format(_spath_remote, e))
            res = False

        return res

    def upload_file(self, _name_bucket: str, _spath_remote: str, _spath_local: str) -> bool:
        self._check_spath_remote(_spath_remote=_spath_remote)

        try:
            self.client.upload_file(_spath_local, _name_bucket, _spath_remote)
            res = True
        except Exception as e:
            print('The uploading a file from {} to {} is failed with the excpetion: {}.'.format(_spath_local, _spath_remote, e))
            res = False

        return res

    def download_file(self, _name_bucket: str, _spath_remote: str, _spath_local: str) -> bool:
        self._check_spath_remote(_spath_remote=_spath_remote)

        try:
            path_local = path_.Path(_spath=_spath_local)
            path_local.unlink(_missing_ok=True)
            path_local.parent.path.mkdir(mode=0o777, parents=True, exist_ok=True)
            self.client.download_file(_name_bucket, _spath_remote, path_local.str)
            res = True
        except Exception as e:
            print('The downloading a file from {} to {} is failed with the excpetion: {}.'.format(_spath_remote, _spath_local, e))
            res = False

        return res

    @staticmethod
    def _check_spath_remote(_spath_remote: str):
        if _spath_remote[0] == '/':
            raise ValueError('The _spath_remote, {} should be start with only userid.'.format(_spath_remote))


class StepFunctions(BaseAWS):
    def __init__(self, _spath_aws: str, _mode: str, _access_key: Optional[str] = None, _secret_key: Optional[str] = None):
        super(StepFunctions, self).__init__(_spath_aws=_spath_aws, _access_key=_access_key, _secret_key=_secret_key)
        self.mode = _mode

        if self.mode not in get_aws_mode_stepfunctions():
            raise NotImplementedError('The AWS step functions mode: {} is not supported yet.'.format(self.mode))

        if _mode in get_aws_mode_base():
            self.client = None
        else:
            self.client = self._get_client(_name_client='stepfunctions')

    def list_state_machines(self, _maxResults: int = 5) -> list:
        response = self.client.list_state_machines(maxResults=_maxResults)

        return response['stateMachines']

    def list_executions(self, _stateMachineArn: str, _statusFilter: str, _maxResults: int = 5) -> list:
        if _statusFilter in {'RUNNING', 'SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED'}:
            response = self.client.list_executions(
                stateMachineArn=_stateMachineArn,
                statusFilter=_statusFilter,
                maxResults=_maxResults
            )
        else:
            raise ValueError('The _statusFilter, {} is not supported yet.'.format(_statusFilter))

        return response['executions']

    def list_executions_history(self, _executionArn: str, _maxResults: int = 5, _reverseOrder: bool = False, _includeExecutionData: bool = True) -> list:
        response = self.client.get_execution_history(
            executionArn=_executionArn,
            maxResults=_maxResults,
            reverseOrder=_reverseOrder,
            includeExecutionData=_includeExecutionData
        )

        return response['events']


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='AWS S3 Boto3')
    parser.add_argument('--mode', '-M', type=str, default='upload', help='Option: awscli; enroll; upload; download; delete.')
    parser.add_argument('--path_remote', '-R', type=str, default='userid/path/object', help='Remote full path (i.e. AWS S3): userid/path/object')
    parser.add_argument('--path_local', '-L', type=str, default='/path/object', help='Local full path: /path/object')
    parser.add_argument('--name_bucket', type=str, default='Secret', help='Bucket name')
    parser.add_argument('--path_aws', type=str, default=os.path.join(str(Path.home()), '.aws'), help='Path for the AWS.')
    parser.add_argument('--region', type=str, default='ap-northeast-2', help='Credentials region')
    parser.add_argument('--output', type=str, default='json', help='Credentials output')

    args = parser.parse_args()

    if args.mode in get_aws_mode_s3():
        printf('args.mode: {}.'.format(args.mode), _is_pause=False)
        aws_s3 = S3(_spath_aws=args.path_aws, _mode=args.mode, _access_key=None, _secret_key=None)
    else:
        raise NotImplementedError('The AWS S3 mode: {} is not supported yet.'.format(args.mode))

    if args.mode == 'upload':
        aws_s3.upload_file(_name_bucket=args.name_bucket, _spath_remote=args.path_remote, _spath_local=args.path_local)
    elif args.mode == 'download':
        aws_s3.download_file(_name_bucket=args.name_bucket, _spath_remote=args.path_remote, _spath_local=args.path_local)
    elif args.mode == 'delete':
        aws_s3.delete_object(_name_bucket=args.name_bucket, _spath_remote=args.path_remote)
    elif args.mode == 'awscli':
        aws_s3.install_awscli()
    elif args.mode == 'enroll':
        aws_s3.enroll_credentials(_spath_aws=args.path_aws, _region=args.region, _output=args.output)
    else:
        raise NotImplementedError('The AWS S3 mode: {} is not supported yet.'.format(args.mode))
