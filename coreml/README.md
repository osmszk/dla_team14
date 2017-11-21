## 環境
- macOS 10.12
- python 2.7(macOS system)
- tensorflow 1.3.0
- Keras 2.0.8
- coremltools 0.6.3

## CoreMLToolsインストールからコンバートまでの手順
`$ virtualenv --python=/usr/bin/python2.7 my_env`

`$ source my_env/bin/activate`

`(my_env) $ pip freeze`

`(my_env) $ pip install -U coremltools`

`(my_env) $ pip install Keras==2.0.8`

`(my_env) $ pip install tensorflow==1.3.0`

`(my_env) $ pip install np_utils`

`(my_env) $ pip install h5py`

`(my_env) $ python convert.py`

## Trouble Shooting
- $ python convet.py で落ちるときは、`$ brew uninstall python` でbrew経由のpythonをアンインストールする。


## Reference
https://stackoverflow.com/questions/44612991/error-installing-coremltools

https://forums.developer.apple.com/thread/80529
