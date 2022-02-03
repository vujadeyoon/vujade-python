"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_aws.py
Description: A module for the Amazon Web Services (AWS)
"""


import os
import sys
import argparse
import json
import boto3
from typing import Set, Optional
from pathlib import Path
from botocore.exceptions import ClientError
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
    def __init__(self, _spath_aws: str, _access_key: Optional[str] = None, _secret_key: Optional[str] = None) -> None:
        super(BaseAWS, self).__init__()
        self.path_aws = path_.Path(_spath=_spath_aws)
        if (_access_key is None) or (_secret_key is None):
            self.access_key, self.secret_key = self._get_aws_access_secret_keys()
        else:
            self.access_key, self.secret_key = _access_key, _secret_key

    def _get_resource(self, _name: str):
        try:
            resource = boto3.resource(_name, aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key)
        except Exception as e:
            raise ConnectionError('It is failed to get a {} resource with the exception: {}.'.format(_name, e))

        return resource

    def _get_client(self, _name: str):
        try:
            client = boto3.client(_name, aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key)
        except Exception as e:
            raise ConnectionError('It is failed to get a {} client with the exception: {}.'.format(_name, e))

        return client

    @staticmethod
    def install_awscli() -> None:
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
    def __init__(self, _mode: str, _access_key: Optional[str] = None, _secret_key: Optional[str] = None, _spath_aws: str = os.path.join(str(Path.home()), '.aws')) -> None:
        super(S3, self).__init__(_spath_aws=_spath_aws, _access_key=_access_key, _secret_key=_secret_key)
        self.mode = _mode

        if self.mode not in get_aws_mode_s3():
            raise NotImplementedError('The AWS S3 mode: {} is not supported yet.'.format(self.mode))

        if _mode in get_aws_mode_base():
            self.client = None
        else:
            self.client = self._get_client(_name='s3')

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
    def __init__(self, _mode: str, _access_key: Optional[str] = None, _secret_key: Optional[str] = None, _spath_aws: str = os.path.join(str(Path.home()), '.aws')) -> None:
        super(StepFunctions, self).__init__(_spath_aws=_spath_aws, _access_key=_access_key, _secret_key=_secret_key)
        self.mode = _mode

        if self.mode not in get_aws_mode_stepfunctions():
            raise NotImplementedError('The AWS step functions mode: {} is not supported yet.'.format(self.mode))

        if _mode in get_aws_mode_base():
            self.client = None
        else:
            self.client = self._get_client(_name='stepfunctions')

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


class SQS(BaseAWS):
    def __init__(self, _access_key: Optional[str] = None, _secret_key: Optional[str] = None, _spath_aws: str = os.path.join(str(Path.home()), '.aws')) -> None:
        super(SQS, self).__init__(_spath_aws=_spath_aws, _access_key=_access_key, _secret_key=_secret_key)
        self.resource = self._get_resource(_name='sqs')
        self.client = self._get_client(_name='sqs')

    def create_queue(self, _name: str, _attributes: Optional[dict] = None, _is_silent: bool = True):
        if _attributes is None:
            _attributes = {}

        try:
            queue = self.resource.create_queue(QueueName=_name, Attributes=_attributes)
            if _is_silent is False:
                print('A queue, {} is created with the url, {}.'.format(_name, queue.url))
        except ClientError as error:
            if _is_silent is False:
                print('A queue, {} cannot be created.'.format(_name))
            raise error

        return queue

    def get_queue(self, _name: str, _is_silent: bool = True):
        try:
            queue = self.resource.get_queue_by_name(QueueName=_name)
            if _is_silent is False:
                print('A queue, {} is gotten with the url, {}.'.format(_name, queue.url))
        except ClientError as error:
            if _is_silent is False:
                print('A queue, {} cannot be gotten.'.format(_name))
            raise error

        return queue

    def remove_queue(self, _name: str, _is_silent: bool = True) -> None:
        queue = self.get_queue(_name=_name)

        try:
            queue.delete()
            if _is_silent is False:
                print('A queue is deleted with the url, {}.'.format(queue.url))
        except ClientError as error:
            if _is_silent is False:
                print('A queue cannot be deleted with the url, {}.'.format(queue.url))
            raise error

    def msg_send(self, _name: str, _msg_body: json, **_kwargs) -> dict:
        # Usage:
        #   i)  Standard: res = sqs.msg_send(_name='name.fifo', _msg_body=json.dumps({'message': "test"}))
        #   ii) FIFO: res = sqs.msg_send(_name='name', _msg_body=json.dumps({'message': "test"}), MessageGroupId='msg_group_id')

        queue = self.get_queue(_name=_name)
        try:
            res = self.client.send_message(QueueUrl=queue.url, MessageBody=_msg_body, **_kwargs)
        except ClientError as e:
            print('It is failed to send a message.')
            res = None

        return res

    def msg_receive(self, _name: str, _is_msg_delete: bool = True) -> dict:
        # Usage: sqs.msg_receive(_name='name', _is_msg_delete=True)

        queue = self.get_queue(_name=_name)
        try:
            res = self.client.receive_message(QueueUrl=queue.url)
            if _is_msg_delete is True:
                self.client.delete_message(QueueUrl=queue.url, ReceiptHandle=res['Messages'][0]['ReceiptHandle'])
        except ClientError as e:
            print('It is failed to receive a message.')
            res = None

        return res

    def get_attributes(self, _name: str) -> dict:
        queue = self.get_queue(_name=_name)
        try:
            res = self.client.get_queue_attributes(QueueUrl=queue.url, AttributeNames=['All'])
        except ClientError as e:
            print('It is failed to get attributes of the queue.')
            res = None

        return res

    def get_num_msg(self, _name: str) -> int:
        queue_attr = self.get_attributes(_name=_name)
        return int(queue_attr['Attributes']['ApproximateNumberOfMessages'])


class DynamoDB(BaseAWS):
    def __init__(self, _access_key: Optional[str] = None, _secret_key: Optional[str] = None, _spath_aws: str = os.path.join(str(Path.home()), '.aws')) -> None:
        """
        Usage:
            param_name_table = 'DynamoDB-Test'
            param_item = {
                'request_id': str(uuid.uuid4()),
                'name_user': 'usrname',
                'key_1': 1,
                'key_2': 2,
                'key_3': 'value_3',
                'key_4': 'value_4'
            }
            param_key = {'request_id': 'a455b494-7fa3-46ce-bc9b-aa562a5db85d', 'name_user': 'usrname'}
            param_update_expression = 'SET key_1=:key_1_replaced, key_2=:100 REMOVE #key_3, #key_4'

            dynamo_db = DynamoDB(_access_key=access_key, _secret_key=secret_key)
            dynamo_db.list_tables()
            dynamo_db.get_table(_name_table=param_name_table)
            dynamo_db.scan(_name_table=param_name_table)
            dynamo_db.create(_name_table=param_name_table, _item=param_item)
            dynamo_db.read(_name_table=param_name_table, _key=param_key)
            dynamo_db.update(_name_table=param_name_table, _key=param_key, _update_expression=param_update_expression)
            dynamo_db.delete(_name_table=param_name_table, _key=param_key)
        """
        super(DynamoDB, self).__init__(_spath_aws=_spath_aws, _access_key=_access_key, _secret_key=_secret_key)
        self.resource = self._get_resource(_name='dynamodb')
        self.client = self._get_client(_name='dynamodb')

    def list_tables(self) -> dict:
        return self.client.list_tables()

    def get_table(self, _name_table: str):
        return self.resource.Table(_name_table)

    def scan(self, _name_table: str) -> dict:
        table = self.get_table(_name_table=_name_table)
        return table.scan()

    def create(self, _name_table: str, _item: dict) -> dict:
        table = self.get_table(_name_table=_name_table)
        return table.put_item(Item=_item)

    def read(self, _name_table: str, _key: dict):
        table = self.get_table(_name_table=_name_table)

        try:
            response = table.get_item(Key=_key)['Item']
        except Exception as e:
            print('[FAIL] DynamoDB-read; Error: {}.'.format(e))
            response = None

        return response

    def update(self, _name_table: str, _key: dict, _update_expression: str, _return_values: str = 'UPDATED_NEW'):
        table = self.get_table(_name_table=_name_table)
        exp_attr_names, exp_attr_values = self._get_params_update(_update_expression=_update_expression)

        kwargns = dict()
        if exp_attr_names is not None:
            kwargns['ExpressionAttributeNames'] = exp_attr_names
        if exp_attr_values is not None:
            kwargns['ExpressionAttributeValues'] = exp_attr_values

        try:
            response = table.update_item(
                Key=_key,
                UpdateExpression=_update_expression,
                ReturnValues=_return_values,
                **kwargns
            )
        except Exception as e:
            print('[FAIL] DynamoDB-update; Error: {}.'.format(e))
            response = None

        return response

    def delete(self, _name_table: str, _key: dict):
        table = self.get_table(_name_table=_name_table)

        try:
            response = table.delete_item(Key=_key)
        except Exception as e:
            print('[FAIL] DynamoDB-delete; Error: {}.'.format(e))
            response = None

        return response

    @classmethod
    def _get_params_update(self, _update_expression: str) -> tuple:
        dict_exp_attr_names = dict()
        dict_exp_attr_values = dict()
        list_exp_attr_names = list()
        list_exp_attr_values = list()
        len_update_expression = len(_update_expression)
        idx_curr = 0

        while idx_curr < len_update_expression:
            char_curr = _update_expression[idx_curr]

            # ExpressionAttributeNames
            if char_curr == '#':
                exp_attr_name = ''
                idy_curr = idx_curr
                while (idy_curr < len_update_expression) and (not _update_expression[idy_curr] in {',', ' '}):
                    exp_attr_name += _update_expression[idy_curr]
                    idy_curr += 1

                list_exp_attr_names.append(exp_attr_name)
                idx_curr = idy_curr

            # ExpressionAttributeValues
            if char_curr == ':':
                exp_attr_val = ''
                idy_curr = idx_curr
                while (idy_curr < len_update_expression) and (not _update_expression[idy_curr] in {',', ' '}):
                    exp_attr_val += _update_expression[idy_curr]
                    idy_curr += 1

                list_exp_attr_values.append(exp_attr_val)
                idx_curr = idy_curr

            idx_curr +=1

        # ExpressionAttributeNames
        if list_exp_attr_names: # Not-empty
            res_exp_attr_names = dict()
            for _idx, _exp_attr_name in enumerate(list_exp_attr_names):
                res_exp_attr_names[_exp_attr_name] = _exp_attr_name[1:]
        else:
            res_exp_attr_names = None

        # ExpressionAttributeValues
        if list_exp_attr_values: # Not-empty
            res_exp_attr_values = dict()
            for _idx, _exp_attr_val in enumerate(list_exp_attr_values):
                res_exp_attr_values[_exp_attr_val] = _exp_attr_val[1:]
        else:
            res_exp_attr_values = None

        return res_exp_attr_names, res_exp_attr_values


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='AWS S3 Boto3')
    parser.add_argument('--mode', '-M', type=str, default='upload', help='Option: awscli; enroll; upload; download; delete.')
    parser.add_argument('--path_remote', '-R', type=str, default='userid/path/object', help='Remote full path (i.e. AWS S3): userid/path/object')
    parser.add_argument('--path_local', '-L', type=str, default='/path/object', help='Local full path: /path/object')
    parser.add_argument('--name_bucket', type=str, default='required', help='Bucket name')
    parser.add_argument('--access_key', type=str, default='required', help='Credentials output')
    parser.add_argument('--secret_key', type=str, default='required', help='Credentials output')
    parser.add_argument('--path_aws', type=str, default=os.path.join(str(Path.home()), '.aws'), help='Path for the AWS.')
    parser.add_argument('--region', type=str, default='ap-northeast-2', help='Credentials region')
    parser.add_argument('--output', type=str, default='json', help='Credentials output')

    args = parser.parse_args()

    if args.mode in get_aws_mode_s3():
        printf('args.mode: {}.'.format(args.mode), _is_pause=False)
        aws_s3 = S3(_mode=args.mode, _access_key=args.access_key, _secret_key=args.secret_key, _spath_aws=args.path_aws)
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
