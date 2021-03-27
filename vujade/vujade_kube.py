"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_kube.py
Description: A module for kubernetes
"""


from kubernetes import client, config
from vujade import vujade_str as str_


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
