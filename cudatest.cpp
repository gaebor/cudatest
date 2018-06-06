// export CUDAROOT=/usr/local/cuda && g++ -O2 -std=c++11 cudatest.cpp -lcublas -lcudart -lpthread -ldl -I $CUDAROOT/include -L $CUDAROOT/lib64 -o cudatest
// cl /O2 /DNDEBUG /EHsc /I"%CUDA_PATH%/include" cudatest.cpp /link /LIBPATH:"%CUDA_PATH%/lib/x64" cudart.lib cublas.lib

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <exception>
#include <random>
#include <chrono>
#include <cmath>

#include "cublas.h"
#include "cublas_v2.h"

#ifndef REAL
#   define REAL float
#endif

#define xstringify(s) stringify(s)
#define stringify(s) #s

const char* cudaError2str(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess: return "Success";
        case cudaErrorMissingConfiguration : return "MissingConfiguration";
        case cudaErrorMemoryAllocation : return "MemoryAllocation";
        case cudaErrorInitializationError : return "InitializationError";
        case cudaErrorLaunchFailure : return "LaunchFailure";
        case cudaErrorPriorLaunchFailure : return "PriorLaunchFailure";
        case cudaErrorLaunchTimeout : return "LaunchTimeout";
        case cudaErrorLaunchOutOfResources : return "LaunchOutOfResources";
        case cudaErrorInvalidDeviceFunction : return "InvalidDeviceFunction";
        case cudaErrorInvalidConfiguration : return "InvalidConfiguration";
        case cudaErrorInvalidDevice : return "InvalidDevice";
        case cudaErrorInvalidValue : return "InvalidValue";
        case cudaErrorInvalidPitchValue : return "InvalidPitchValue";
        case cudaErrorInvalidSymbol : return "InvalidSymbol";
        case cudaErrorMapBufferObjectFailed : return "MapBufferObjectFailed";
        case cudaErrorUnmapBufferObjectFailed : return "UnmapBufferObjectFailed";
        case cudaErrorInvalidHostPointer : return "InvalidHostPointer";
        case cudaErrorInvalidDevicePointer : return "InvalidDevicePointer";
        case cudaErrorInvalidTexture : return "InvalidTexture";
        case cudaErrorInvalidTextureBinding : return "InvalidTextureBinding";
        case cudaErrorInvalidChannelDescriptor : return "InvalidChannelDescriptor";
        case cudaErrorInvalidMemcpyDirection : return "InvalidMemcpyDirection";
        case cudaErrorAddressOfConstant : return "AddressOfConstant";
        case cudaErrorTextureFetchFailed : return "TextureFetchFailed";
        case cudaErrorTextureNotBound : return "TextureNotBound";
        case cudaErrorSynchronizationError : return "SynchronizationError";
        case cudaErrorInvalidFilterSetting : return "InvalidFilterSetting";
        case cudaErrorInvalidNormSetting : return "InvalidNormSetting";
        case cudaErrorMixedDeviceExecution : return "MixedDeviceExecution";
        case cudaErrorCudartUnloading : return "CudartUnloading";
        case cudaErrorUnknown : return "Unknown";
        case cudaErrorNotYetImplemented : return "NotYetImplemented";
        case cudaErrorMemoryValueTooLarge : return "MemoryValueTooLarge";
        case cudaErrorInvalidResourceHandle : return "InvalidResourceHandle";
        case cudaErrorNotReady : return "NotReady";
        case cudaErrorInsufficientDriver : return "InsufficientDriver";
        case cudaErrorSetOnActiveProcess : return "SetOnActiveProcess";
        case cudaErrorInvalidSurface : return "InvalidSurface";
        case cudaErrorNoDevice : return "NoDevice";
        case cudaErrorSharedObjectSymbolNotFound : return "SharedObjectSymbolNotFound";
        case cudaErrorSharedObjectInitFailed : return "SharedObjectInitFailed";
        case cudaErrorUnsupportedLimit : return "UnsupportedLimit";
        case cudaErrorDuplicateVariableName : return "DuplicateVariableName";
        case cudaErrorDuplicateTextureName : return "DuplicateTextureName";
        case cudaErrorDuplicateSurfaceName : return "DuplicateSurfaceName";
        case cudaErrorDevicesUnavailable : return "DevicesUnavailable";
        case cudaErrorInvalidKernelImage : return "InvalidKernelImage";
        case cudaErrorNoKernelImageForDevice : return "NoKernelImageForDevice";
        case cudaErrorIncompatibleDriverContext : return "IncompatibleDriverContext";
        case cudaErrorPeerAccessAlreadyEnabled : return "PeerAccessAlreadyEnabled";
        case cudaErrorPeerAccessNotEnabled : return "PeerAccessNotEnabled";
        case cudaErrorDeviceAlreadyInUse : return "DeviceAlreadyInUse";
        case cudaErrorProfilerDisabled : return "ProfilerDisabled";
        case cudaErrorProfilerNotInitialized : return "ProfilerNotInitialized";
        case cudaErrorProfilerAlreadyStarted : return "ProfilerAlreadyStarted";
        case cudaErrorProfilerAlreadyStopped : return "ProfilerAlreadyStopped";
        case cudaErrorAssert : return "Assert";
        case cudaErrorTooManyPeers : return "TooManyPeers";
        case cudaErrorHostMemoryAlreadyRegistered : return "HostMemoryAlreadyRegistered";
        case cudaErrorHostMemoryNotRegistered : return "HostMemoryNotRegistered";
        case cudaErrorOperatingSystem : return "OperatingSystem";
        case cudaErrorPeerAccessUnsupported : return "PeerAccessUnsupported";
        case cudaErrorLaunchMaxDepthExceeded : return "LaunchMaxDepthExceeded";
        case cudaErrorLaunchFileScopedTex : return "LaunchFileScopedTex";
        case cudaErrorLaunchFileScopedSurf : return "LaunchFileScopedSurf";
        case cudaErrorSyncDepthExceeded : return "SyncDepthExceeded";
        case cudaErrorLaunchPendingCountExceeded : return "LaunchPendingCountExceeded";
        case cudaErrorNotPermitted : return "NotPermitted";
        case cudaErrorNotSupported : return "NotSupported";
        case cudaErrorHardwareStackError : return "HardwareStackError";
        case cudaErrorIllegalInstruction : return "IllegalInstruction";
        case cudaErrorMisalignedAddress : return "MisalignedAddress";
        case cudaErrorInvalidAddressSpace : return "InvalidAddressSpace";
        case cudaErrorInvalidPc : return "InvalidPc";
        case cudaErrorIllegalAddress : return "IllegalAddress";
        case cudaErrorInvalidPtx : return "InvalidPtx";
        case cudaErrorInvalidGraphicsContext : return "InvalidGraphicsContext";
        case cudaErrorNvlinkUncorrectable : return "NvlinkUncorrectable";
        case cudaErrorApiFailureBase : return "ApiFailureBase";
        default : return "Unknown";
    }
}

const char* cublasStatus2str(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "LICENSE_ERROR";
        default: return "UNKOWN";
    }
}

class CudaException : public std::exception
{
public:
    CudaException(cudaError_t e, const char* m)
    : error(e), msg(m)
    {
        assign();
    }
    CudaException& operator= (const CudaException& other)throw();
    virtual ~CudaException() throw() {}
    virtual const char* what() const throw()
    {
        return message.c_str();
    }
private:
    void assign()
    {
        std::ostringstream oss;
        oss << "CUDA error " << error << " (" << cudaError2str(error) <<
               ") in \"" << msg << "\"";
        message = oss.str();
    }
    cudaError_t error;
protected:
    const char* msg;
    std::string message;
};

class CuBlasException : public CudaException
{
public:
    CuBlasException(cublasStatus_t e, const char* m)
    : CudaException(cudaSuccess, m), status(e)
    {
        assign();
    }
    virtual ~CuBlasException() throw() {}
    virtual const char* what() const throw()
    {
        return message.c_str();
    }
private:
    void assign()
    {
        std::ostringstream oss;
        oss << "cuBLAS error " << status << " (" << cublasStatus2str(status) <<
               ") in \"" << msg << "\"";
        message = oss.str();
    }
    cublasStatus_t status;
};

#define cuda_run(X, ...) do { \
        cudaError_t error = X(__VA_ARGS__); \
        if (error != cudaSuccess) throw CudaException(error, #X); } \
        while (false)

#define cublas_run(X, ...) do { \
        cublasStatus_t error = X(__VA_ARGS__); \
        if (error != CUBLAS_STATUS_SUCCESS) throw CuBlasException(error, #X); } \
        while (false)

template<class real>
real dot(const real* x, const real* y, int n)
{
    real result;
    return result;
}

cublasHandle_t handle;
bool verbose = false;

template<>
float dot(const float* x, const float* y, int n)
{
    float result;
    cublas_run(cublasSdot, handle, n, x, 1, y, 1, &result);
    return result;
}

template<>
double dot(const double* x, const double* y, int n)
{
    double result;
    cublas_run(cublasDdot, handle, n, x, 1, y, 1, &result);
    return result;
}

int main(int argc, char* argv[])
{
    bool use_random = false;
    size_t upper_bound = 0;
    bool cuda_alloc = false;
try
{
    for (++argv; *argv; ++argv)
    {
        if (strcmp("--random", *argv) == 0 || strcmp("-r", *argv) == 0)
            use_random = true;
        else if (strcmp("--verbose", *argv) == 0 || strcmp("-v", *argv) == 0)
            verbose = true;
        else if (strcmp("-c", *argv) == 0 || strcmp("--cuda", *argv) == 0)
            cuda_alloc = true;
        else if (
                    (strcmp("--upper_bound", *argv) == 0 ||
                    strcmp("--upper-bound", *argv) == 0 ||
                    strcmp("-U", *argv) == 0
                    ) && *(argv+1))
            upper_bound = atoi(*++argv);
        else if (strcmp("--help", *argv) == 0 || strcmp("-h", *argv) == 0)
        {
            int cuda_version;
            cuda_run(cudaDriverGetVersion, &cuda_version);
            std::cout << "CUDA (driver v" << cuda_version  << ", runtime v";
            cuda_run(cudaRuntimeGetVersion, &cuda_version);
            std::cout << cuda_version << ") test app.\n"
"Written by Gabor Borbely, contact: borbely@math.bme.hu\n\n"
"The app\n"
" - allocates a certain amount of memory\n"
" - transfers it to the device\n"
" - performs a simple BLAS calculation (" xstringify(REAL) " precision)\n"
" - then frees the allocated memories.\n"
"Also tries to push the allocated amount as high as possible.\n"
"Does this for all the GPUs CUDA can find.\n\n"
"OPTIONS:\n"
"-r --random\tfills the test vector with random elements.\n"
"-v --verbose\tprints more info to stderr.\n"
"-U --upper-bound <int>\tdetermines the maximum length of the vector to allocate.\n"
"-c --cuda\tuse cuda api for allocating memory even on host.\n"
"\nThe columns of stdout (tab separated):\n"
"Device (Compute major.minor)\n"
"Memory in bytes\n"
"Memory clock rate\n"
"Multiprocessor count\nProcessor clock rate\n"
"Total amount of memory that was successfully allocated and transfered to device\n"
            <<std::endl;
            return 0;
        }
        else
            std::cerr << "unknown parameter \"" << *argv << "\"" << std::endl;
    }
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<REAL> distribution(-1.0,1.0);
    
    int device_count;
    cuda_run(cudaGetDeviceCount, &device_count);

    size_t count;
    size_t m;
    size_t succeeded;
    
    for (int device=0; device < device_count; ++device)
    {
        struct cudaDeviceProp prop;
        
        cuda_run(cudaSetDevice, device);
        cuda_run(cudaGetDeviceProperties, &prop, device);
        printf("%s (%d.%d)\t", prop.name, prop.major, prop.minor);
        std::cout << prop.totalGlobalMem;
        printf("\t@%gGHz\tx%d\t@%gGHz\t", prop.memoryClockRate/1000000.0,
                                  prop.multiProcessorCount, prop.clockRate/1000000.0);
        
        cublas_run(cublasCreate, &handle);
        
        count = 0;
        
        size_t u;
        if (upper_bound > 0)
            u = upper_bound;
        else
            u = std::pow(2.0, std::ceil(std::log2(prop.totalGlobalMem/sizeof(REAL))));
        for ( ; u > 0; u /= 2)
        {
            try
            {
                count += u;
                m = sizeof(REAL)*count;
                REAL* x_host, *x_dev;
                
                if (cuda_alloc)
                    cuda_run(cudaMallocHost, &x_host, m);
                else
                {
                    if (verbose)
                        std::cerr << "malloc\t"; std::cerr.flush();
                    x_host = (REAL*)malloc(m);
                    if (!x_host)
                       throw CudaException(cudaSuccess, "malloc");
                }
                if (use_random)
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        x_host[i] = distribution(generator);
                    }
                }
                
                cuda_run(cudaMalloc, &x_dev, m);

                cuda_run(cudaMemcpy, x_dev, x_host, m, cudaMemcpyHostToDevice);
                
                dot(x_dev, x_dev, count);
                std::cerr << m << "\t";
                std::cerr.flush();
                
                cuda_run(cudaFree, x_dev);

                if (cuda_alloc)
                    cuda_run(cudaFreeHost, x_host);
                else
                    free(x_host);
            }
            catch (CudaException& e)
            {   //unsuccessful
                std::cerr << e.what() << "\t";
                std::cerr.flush();
                count -= u;
                continue;
            }
        }
        cublas_run(cublasDestroy, handle);
        std::cout << count*sizeof(REAL) << std::endl;
        std::cerr << std::endl;
    }
}
    catch (CudaException& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
