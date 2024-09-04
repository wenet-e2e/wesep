# Libtorch backend on wesep

* Build. The build requires cmake 3.14 or above, and gcc/g++ 5.4 or above.

``` sh
mkdir build && cd build
cmake ..
cmake --build .
```

* Testing.

1. the RTF(real time factor) is shown in the console, and outputs will be written to the wav file.

``` sh
export GLOG_logtostderr=1
export GLOG_v=2

./build/bin/separate_main \
  --wav_scp $wav_scp \
  --model /path/to/model.zip \
  --output_dir /output/dir/
```
