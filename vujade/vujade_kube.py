"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_kube.py
Description: A module for kubernetes
"""


from kubernetes import client, config
from vujade import vujade_str as str_
from vujade import vujade_path as path_


class KubeEnvVariables(object):
    def __init__(self, _spath_configmap: str, _spath_secret: str) -> None:
        super(KubeEnvVariables, self).__init__()
        self.path_configmap = path_.Path(_spath_configmap)
        self.path_secret = path_.Path(_spath_secret)

    def get_configmap(self, _is_print: bool = False) -> dict:
        res = dict()
        try:
            for _idx, _path_cfg_item in enumerate(self.path_configmap.path.glob('[!..]*')):
                with open(str(_path_cfg_item), 'r') as f:
                    res[_path_cfg_item.name] = f.readline()
            if _is_print is True:
                print('The config map: {}'.format(res))
        except Exception as e:
            print('It is failed to get config map values; Exception: {}'.format(e))

        return res

    def get_secret(self, _is_print: bool = False) -> dict:
        res = dict()
        try:
            for _idx, _path_secret_item in enumerate(self.path_secret.path.glob('[!..]*')):
                with open(str(_path_secret_item), 'r') as f:
                    res[_path_secret_item.name] = f.readline().rstrip('\n')
            if _is_print is True:
                print('The secret: {}'.format(res))
        except Exception as e:
            print('It is failed to get secret values; Exception: {}'.format(e))

        return res


def get_pod(_ip_adrr: str, _namespace: str, _watch: bool = False) -> str:
    try:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        ret = v1.list_namespaced_pod(namespace=_namespace, watch=_watch)

        res = 'unknown'
        for i in ret.items:
            if _ip_adrr == i.status.pod_ip:
                res = i.metadata.name
    except:
        res = 'unknown'

    return res


def get_node_pod(_ip_adrr: str, _namespace: str, _watch: bool = False) -> tuple:
    try:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        ret = v1.list_namespaced_pod(namespace=_namespace, watch=_watch)

        name_node = 'unknown'
        name_pod = 'unknown'
        for i in ret.items:
            if _ip_adrr == i.status.pod_ip:
                name_pod = i.metadata.name

                spec = str(i.spec)
                dict_spec = str_.str2dict_2(_str=spec)
                name_node = dict_spec["'node_name':"].replace(',', '').replace("'", '')
    except:
        name_node = 'unknown'
        name_pod = 'unknown'

    return name_node, name_pod


def get_running_pod_name(_pod_prefix: str, _namespace: str, _watch: bool = False) -> list:
    res = []

    config.load_kube_config()
    v1 = client.CoreV1Api()
    ret = v1.list_namespaced_pod(namespace=_namespace, watch=_watch)

    for i in ret.items:
        pod_name = i.metadata.name
        pod_status = i.status.phase

        if (_pod_prefix in pod_name) and (pod_status == 'Running'):
            res.append(pod_name)

    return res
