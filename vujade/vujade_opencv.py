"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_opencv.py
Description: A module for C++ based OpenCV
"""


import numpy as np
from vujade import vujade_json as json_


'''
C++ code:

class CppOpenCVMat {
private:
    const char* m_path_filename;

public:
    cv::Mat run_import(const char* name_mat = "cameraMatrix") {
        cv::Mat cameraMatrix;
        cv::FileStorage fs(m_path_filename, cv::FileStorage::READ);
        if (fs.isOpened()) {
            fs["cameraMatrix"] >> cameraMatrix;
        }
        else {
            Log::w("It is failed to open the file pointer: ", m_path_filename);
        }
        fs.release();
        return cameraMatrix;
    }

    void run_export(cv::Mat& mat, const char* name_mat = "cameraMatrix") {
        cv::FileStorage fs(m_path_filename, cv::FileStorage::WRITE);
        fs << name_mat << mat;
        fs.release();
    }

    CppOpenCVMat(const char* path_filename = "/sdcard/Download/mat.json") {
        m_path_filename = path_filename;
    }

    ~CppOpenCVMat() {
    }
};
'''

class CppOpenCVMat(object):
    """
    json format for cv::mat:
        {
            "cameraMatrix": {
                "type_id": "opencv-matrix",
                "rows": 3,
                "cols": 1,
                "dt": "3f",
                "data": [ 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0 ] # [ B, G, R, B, G, R, B, G, R ]
            }
        }
    """
    @staticmethod
    def run_import(_spath_filename: str, _name_ndarr: str = 'cameraMatrix') -> np.ndarray:
        # Import a json file into a ndarray for cv::Mat based on C++ OpenCV
        data = json_.JSON.read(_spath_filename=_spath_filename, _mode='r')
        data_mat = data[_name_ndarr]
        ndarr_rows = data_mat['rows']
        ndarr_cols = data_mat['cols']
        ndarr_dt = data_mat['dt']
        ndarr_channels = 1 if ndarr_dt[:-1] == '' else int(ndarr_dt[:-1])
        ndarr_data = np.asarray(data_mat['data']).reshape((ndarr_rows, ndarr_cols, ndarr_channels)).astype(np.float32)

        return ndarr_data

    @staticmethod
    def run_export(_spath_filename: str, _ndarr: np.ndarray, _name_ndarr: str = 'cameraMatrix') -> None:
        # Export a ndarray into a json file for cv::Mat based on C++ OpenCV
        ndarr_rows = _ndarr.shape[0]
        ndarr_cols = _ndarr.shape[1]
        if _ndarr.ndim == 2:
            ndarr_channels = 1
        elif _ndarr.ndim == 3:
            ndarr_channels = _ndarr.shape[2]
        else:
            raise NotImplementedError

        data = {
            _name_ndarr: {
                'type_id': 'opencv-matrix',
                'rows': ndarr_rows,
                'cols': ndarr_cols,
                'dt': 'f' if ndarr_channels == 1 else '{}f'.format(ndarr_channels),
                "data": _ndarr.astype(np.float32).flatten().tolist()
            }
        }

        json_.JSON.write(_spath_filename=_spath_filename, _data=data)
