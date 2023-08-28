"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_tensorflow.py
Description: A module for TensorFlow
"""


import onnx
import tflite_runtime
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from typing import List, Optional
from onnx_tf.backend import prepare
from vujade import vujade_path as path_
from vujade import vujade_utils as utils_
from vujade.vujade_debug import printd


class TensorFlow(object):
    def __init__(self) -> None:
        super(TensorFlow, self).__init__()

    @staticmethod
    def onnx2tf(_spath_dir_tf: str, _model_onnx: onnx.onnx_ml_pb2.ModelProto) -> None:
        path_dir_tf = path_.Path(_spath_dir_tf)

        path_dir_tf.path.mkdir(mode=0o755, parents=True, exist_ok=True)
        tf_rep = prepare(_model_onnx)
        tf_rep.export_graph(path_dir_tf.str)

    @staticmethod
    def load(_spath_dir_tf: str, _is_trainable: bool = False):
        model_tf = tf.saved_model.load(_spath_dir_tf)
        model_tf.trainable = _is_trainable
        return model_tf

    @staticmethod
    def run(_ndarr_input: np.ndarray, _model_tf) -> dict:
        input_tensor = tf.convert_to_tensor(_ndarr_input)
        return _model_tf(**{'input': input_tensor})


class TensorFlowLite(object):
    def __init__(self, _spath_tflite: str, _experimental_delegates: Optional[list] = None, _num_threads: int = 1) -> None:
        super(TensorFlowLite, self).__init__()
        self.path_tflite = path_.Path(_spath_tflite)
        self.experimental_delegates = _experimental_delegates
        self.num_threads = _num_threads
        self.interpreter = self._get_interpreter()
        self._allocate_tensor()
        self.details_inputs, self.details_outputs = self._get_details()
        self.num_inputs, self.num_outputs = len(self.details_inputs), len(self.details_outputs)

    def run(self, _ndarr_inputs: List[np.ndarray]) -> List[np.ndarray]:
        self._set_tensor_inputs(_ndarr_inputs=_ndarr_inputs)
        self._invoke()
        return self._get_tensor_outputs()

    def get_details(self, _name_key: str = 'inputs') -> dict:
        res = dict()
        name_keys = {'inputs', 'outputs'}

        if _name_key == 'inputs':
            details = self.details_inputs
        elif _name_key == 'outputs':
            details = self.details_outputs
        else:
            raise ValueError('The given _name_key (i.e. {}) should be in {}.'.format(_name_key, name_keys))

        for _idx, _detail in enumerate(details):
            res.update({_idx: _detail})

        return res

    def _get_interpreter(self) -> tflite_runtime.interpreter.Interpreter:
        return tflite.Interpreter(
            model_path=self.path_tflite.str,
            experimental_delegates=self.experimental_delegates,
            num_threads=self.num_threads
        )

    def _allocate_tensor(self) -> None:
        self.interpreter.allocate_tensors()

    def _get_details(self) -> tuple:
        return self.interpreter.get_input_details(), self.interpreter.get_output_details()

    def _set_tensor_inputs(self, _ndarr_inputs: List[np.ndarray]) -> None:
        for _idx, _ndarr_input in enumerate(_ndarr_inputs):
            self._set_tensor_input(_ndarr_input=_ndarr_input, _idx=_idx)

    def _get_tensor_outputs(self) -> List[np.ndarray]:
        res = [self._get_tensor_output(_idx) for _idx in range(self.num_outputs)]
        return res

    def _set_tensor_input(self, _ndarr_input: np.ndarray, _idx: int = 0) -> None:
        self.interpreter.set_tensor(self.details_inputs[_idx]['index'], _ndarr_input)

    def _get_tensor_output(self, _idx: int = 0) -> np.ndarray:
        return self.interpreter.get_tensor(self.details_outputs[_idx]['index'])

    def _invoke(self) -> None:
        self.interpreter.invoke()

    @classmethod
    def tf2tflite(cls, _spath_dir_tf: str) -> bytes:
        converter = tf.lite.TFLiteConverter.from_saved_model(_spath_dir_tf)
        model_tflite = converter.convert()

        return model_tflite

    @staticmethod
    def save(_spath_tflite: str, _model_tflite: bytes) -> None:
        with open(_spath_tflite, 'wb') as f:
            f.write(_model_tflite)

    @classmethod
    def is_gpu_compatibility(cls, _model_tflite: bytes) -> bool:
        str_analyzed = cls.analyze(_model_tflite=_model_tflite, _is_gpu_compatibility=True, _is_print=False)
        if 'GPU COMPATIBILITY WARNING' not in str_analyzed:
            res = True
        else:
            res = False

        return res

    @staticmethod
    def analyze(_model_tflite: bytes, _is_gpu_compatibility: bool = True, _is_print: bool = True) -> str:
        utils_.CapturePrint.lock()
        tf.lite.experimental.Analyzer.analyze(model_content=_model_tflite, gpu_compatibility=_is_gpu_compatibility)
        str_captured = utils_.CapturePrint.run(_is_unlock=True)
        if _is_print is True:
            print(str_captured)
        return str_captured
