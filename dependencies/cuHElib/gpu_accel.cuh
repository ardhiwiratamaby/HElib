#ifndef GPU_ACCEL_H
#define GPU_ACCEL_H

#include <helib/NumbTh.h>
#include <helib/timing.h>
#include "cuda_runtime.h"
#include <cufft.h>

#define THREADS_PER_BLOCK 1024

static const char* _cudaGetErrorEnum(cufftResult error)
{
    if (error == CUFFT_SUCCESS)
    {
        return "CUFFT_SUCCESS";
    }
    else if (error == CUFFT_INVALID_PLAN)
    {
        return "CUFFT_INVALID_PLAN";
    }
    else if (error == CUFFT_ALLOC_FAILED)
    {
        return "CUFFT_ALLOC_FAILED";
    }
    else if (error == CUFFT_INVALID_TYPE)
    {
        return "CUFFT_INVALID_TYPE";
    }
    else if (error == CUFFT_INVALID_VALUE)
    {
        return "CUFFT_INVALID_VALUE";
    }
    else if (error == CUFFT_INTERNAL_ERROR)
    {
        return "CUFFT_INTERNAL_ERROR";
    }
    else if (error == CUFFT_EXEC_FAILED)
    {
        return "CUFFT_EXEC_FAILED";
    }
    else if (error == CUFFT_SETUP_FAILED)
    {
        return "CUFFT_SETUP_FAILED";
    }
    else if (error == CUFFT_INVALID_SIZE)
    {
        return "CUFFT_INVALID_SIZE";
    }
    else if (error == CUFFT_UNALIGNED_DATA)
    {
        return "CUFFT_UNALIGNED_DATA";
    }
    else
    {
        return "<unknown>";
    }
}


#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}
static const char *cuFFTCheck(cufftResult error);

#define CHECK_CUFFT_ERRORS(call) { \
    cufftResult_t err; \
    if ((err = (call)) != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", err, _cudaGetErrorEnum(err), \
                __FILE__, __LINE__); \
        exit(1); \
    } \
}
void InitGPUBuffer(long phim, int n_rows, long *d_A, long *d_B, long *d_C, long *d_modulus, long *d_scalar, int bytes);
void DestroyGPUBuffer();
unsigned long long bitReverse(unsigned long long a, int bit_length);  // reverses the bits for twiddle factor calculation
unsigned long long modpow64(unsigned long long a, unsigned long long b, unsigned long long mod);
inline uint64_t Log2(uint64_t x);

void setRowMapA(long offset, long *source, long *contiguousHostMapA, long d_phim);
void setRowMapB(long offset, const long *source, long *contiguousHostMapB, long d_phim);
long *getRowMapB(long index, long *contiguousHostMapB);
long *getRowMapA(long index, long *contiguousHostMapA);

void setMapA(long index, long data, long *contiguousHostMapA);
void setMapB(long index, long data, long *contiguousHostMapB);
void setModulus(long index, long data, long *contiguousModulus, long n_rows);
void setScalar(long index, long data, long *scalarPerRow);

long getMapA(long index, long *contiguousHostMapA);
long getMapB(long index, long *contiguousHostMapB);

void InitContiguousHostMapModulus(long phim, int n_rows, long *contiguousHostMapA, long *contiguousHostMapB, long *contiguousModulus, long *scalarPerRow);

void CudaEltwiseAddMod(long actual_nrows, long *contiguousHostMapA, long *contiguousHostMapB, long *contiguousModulus, long *d_A, long *d_B, long *d_C, long *d_modulus, int bytes, long d_phim);
void CudaEltwiseAddMod(long actual_nrows, long scalar, long *contiguousHostMapA, long *scalarPerRow, long *contiguousModulus, long *d_A, long *d_C, long *d_scalar, long *d_modulus, int bytes, long d_phim);
void CudaEltwiseSubMod(long actual_nrows, long *contiguousHostMapA, long *contiguousHostMapB, long *contiguousModulus, long *d_A, long *d_B, long *d_C, long *d_modulus, int bytes, long d_phim);
void CudaEltwiseSubMod(long actual_nrows, long scalar, long *contiguousHostMapA, long *scalarPerRow, long *contiguousModulus, long *d_A, long *d_C, long *d_scalar, long *d_modulus, int bytes, long d_phim);
void CudaEltwiseMultMod(long actual_nrows, long *contiguousHostMapA, long *contiguousHostMapB, long *contiguousModulus, long *d_A, long *d_B, long *d_C, long *d_modulus, int bytes, long d_phim);
void CudaEltwiseMultMod(long actual_nrows, long scalar, long *contiguousHostMapA, long *scalarPerRow, long *contiguousModulus, long *d_A, long *d_C, long *d_scalar, long *d_modulus, int bytes, long d_phim);

int cuda_add();
void init_gpu_ntt(unsigned int n);
void moveTwFtoGPU(unsigned long long gpu_powers_dev[], std::vector<unsigned long long>& gpu_powers, int k2, NTL::zz_pX& powers, unsigned long long gpu_powers_m_dev[], long zMStar_dev[], long zMStar_h[], long mm, long target_dev[], long target_h[]);
void gpu_mulMod(NTL::zz_pX& x, unsigned long long x_dev[], unsigned long long gpu_powers_m_dev[], unsigned long long p, int n, cudaStream_t stream);
void gpu_mulMod2(NTL::zz_pX& x, unsigned long long x_dev[], unsigned long long x_pinned[], unsigned long long gpu_powers_m_dev[], unsigned long long p, int n, cudaStream_t stream);

void gpu_ntt(unsigned int n, NTL::zz_pX& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv);
void gpu_ntt(unsigned long long res[], unsigned int n, const NTL::zz_pX& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse);
void gpu_ntt(unsigned long long res[], unsigned int n, unsigned long long x[], unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse);
void gpu_ntt(NTL::vec_zz_p& res, unsigned int n, const NTL::zz_pX& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse);
void gpu_ntt(NTL::vec_zz_p& res, unsigned int n, const NTL::vec_zz_p& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse);

void gpu_ntt_forward(unsigned long long res[], unsigned int n, const NTL::zz_pX& x, unsigned long long q, const std::vector<unsigned long long>& gpu_powers, unsigned long long psi, unsigned long long psiinv);
void gpu_ntt_backward(NTL::vec_zz_p& res, unsigned int n, const NTL::vec_zz_p& x, unsigned long long q, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv);
void gpu_fused_polymul(NTL::vec_zz_p& res, unsigned long long a_dev[], const unsigned long long b_dev[], int n, unsigned long long n_inv, unsigned long long x_dev[], unsigned long long q, const std::vector<unsigned long long>& gpu_powers, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv, int l, unsigned long long gpu_powers_dev[], unsigned long long gpu_ipowers_dev[], cudaStream_t stream);
void gpu_ntt_forward_old(NTL::vec_zz_p& res, unsigned int n, const NTL::zz_pX& x, unsigned long long q, const std::vector<unsigned long long>& gpu_powers, unsigned long long psi, unsigned long long psiinv);
void gpu_ntt_backward_old(NTL::vec_zz_p& res, unsigned int n, const NTL::vec_zz_p& x, unsigned long long q, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv);
void gpu_addMod(unsigned long long x_dev[], long n, long dx, unsigned long long p, cudaStream_t stream);
void gpu_parallel_copy(long m, unsigned long long *x_pinned, unsigned long long *x_dev, long *zMStar_gpu, NTL::vec_long& y_h, long target_dev[], cudaStream_t stream);

#if 1//Ardhi: blockingFunction
void gpu_fused_polymul(NTL::vec_zz_p& res, unsigned long long a_dev[], const unsigned long long b_dev[], int n, unsigned long long n_inv, unsigned long long x_dev[], unsigned long long q, const std::vector<unsigned long long>& gpu_powers, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv, int l, unsigned long long gpu_powers_dev[], unsigned long long gpu_ipowers_dev[]);
void gpu_addMod(unsigned long long x_dev[], long n, long dx, unsigned long long p);
void gpu_parallel_copy(long m, unsigned long long *x_pinned, unsigned long long *x_dev, long *zMStar_gpu, NTL::vec_long& y_h, long target_dev[]);
void gpu_mulMod(NTL::zz_pX& x, unsigned long long x_dev[], unsigned long long gpu_powers_m_dev[], unsigned long long p, int n);
void gpu_mulMod2(NTL::zz_pX& x, unsigned long long x_dev[], unsigned long long x_pinned[], unsigned long long gpu_powers_m_dev[], unsigned long long p, int n);
#endif

void initializeStreams(long n_streams, std::vector<cudaStream_t> &streams);
void usecuFFT(std::vector<std::complex<double>>& buf, long m, const cufftHandle& plan, cufftDoubleComplex *buf_dev);
void initcuFFTBuffer(long m, cufftHandle plan, cufftDoubleComplex *buf_dev);

#endif