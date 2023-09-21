<h1 align="center">
<img src="asset/logo/primary/logo.png" width="800">
</h1><br>


[![PyPI](https://img.shields.io/pypi/v/vujade?label=pypi%20package)](https://pypi.org/project/vujade/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/vujade.svg?label=PyPI%20downloads)](
https://pypi.org/project/vujade/)



## Table of contents
1. [Notice](#notice)
2. [How to install Vujade for Python](#how_to_install)
3. [How to add project path to the PYTHONPATH](#export_pythonpath)
4. [How to remove ^M characters](#remove_^M)
5. [License](#license)
6. [Todo](#todo)
7. [Reference](#ref)


## 1. Notice <a name="notice"></a>
- A collection of useful classes and functions based on the Python3 for deep learning research and development
    - The vujade consists of useful codes that I coded myself, wrapper classes and wrapper functions for the Python3 package.
    - Once you're comfortable with vujade, you don't need to search the internet to see how to use functions and classes.
- I recommend that you should ignore the commented instructions with an octothorpe, #.


## 2. How to install Vujade for Python <a name="how_to_install"></a>
- Please note that some Python3 packages should be required. You can install the rquired packages using Python3 official package manager, pip3.
- Please note that the pip3 based vujade has not fully supported all features yet.
### 1. How to install vujade using pip3 from the PyPI
```bash
$ pip3 install vujade
```

### 2. How to install vujade using pip3 from the GitHub repository
```bash
$ pip3 install git+https://github.com/vujadeyoon/vujade-python@${name_tag}
```

### 3. How to install vujade using pip3 from local repository
```bash
$ pip3 install -e .
```

### 4. How to merge vujade
```bash
$ git clone https://github.com/vujadeyoon/vujade-python vujade-python
$ cd ./vujade-python/ && bash ./script/bash_setup_vujade.sh && cd ../
```


## 3. How to add project path to the PYTHONPATH  <a name="export_pythonpath"></a>
If some errors occur for the vujade path in the project, I recommend you should add the project path to the PYTHONPATH.
```bash
$ export PYTHONPATH=$PYTHONPATH:$(pwd)
```


## 4. How to remove ^M characters <a name="remove_^M"></a>
### 1. dos2unix
  ```bash
  $ dos2unix ${PATH_FILE}
  ```

### 2. Sed
```bash
$ sed 's/^M//g' ${PATH_FILE}
```


## 5. License <a name="license"></a>
- I respect and follow the license of the used libraries including python3 packages.
- The libraries including python3 packages are licensed by their licenses.
- Please note that the only providen vujadeyoon's own codes and wrapper-codes comply with the MIT license.


## 6. Todo <a name="todo"></a>
- Instructions for usage will be updated in the future.


## 7. Reference <a name="ref"></a>
1. [remove ^M characters from file using sed](https://stackoverflow.com/questions/19406418/remove-m-characters-from-file-using-sed)
