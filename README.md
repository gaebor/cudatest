# cudatest
tests the cuda api

## Dependencies
* https://developer.nvidia.com/cuda-downloads
* c++11 compliant compiler

## Compile
See the first line of `cudatest.cpp`
* Linux (here CUDA is installed in `/usr/local/cuda`)

      export CUDAROOT=/usr/local/cuda && g++ -O2 -std=c++11 cudatest.cpp -lcublas -lcudart -lpthread -ldl -I $CUDAROOT/include -L $CUDAROOT/lib64 -o cudatest

* Windows: `CUDA_PATH` environment variable have to be set (it is if you have installed CUDA Toolkit)

      cl /O2 /DNDEBUG /EHsc /I"%CUDA_PATH%/include" cudatest.cpp /link /LIBPATH:"%CUDA_PATH%/lib/x64" cudart.lib cublas.lib

By default it is compiled to test `float` precision, but you can redefine it.
Compile with this macro to get double precision:

    DREAL=double

## Usage
Just run `cudatest` or you can see the help: `cudatest -h`.

If everything works fine, just omit the errors:

    cudatest 2> /dev/null
