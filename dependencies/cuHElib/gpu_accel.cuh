#ifndef GPU_ACCEL_H
#define GPU_ACCEL_H

#include <helib/NumbTh.h>
#include <helib/timing.h>

#define THREADS_PER_BLOCK 1024

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

void InitGPUBuffer(long phim, int n_rows);
void DestroyGPUBuffer();
unsigned long long bitReverse(unsigned long long a, int bit_length);  // reverses the bits for twiddle factor calculation
unsigned long long modpow64(unsigned long long a, unsigned long long b, unsigned long long mod);
inline uint64_t Log2(uint64_t x);

void setRowMapA(long offset, long *source);
void setRowMapB(long offset, const long *source);
long *getRowMapB(long index);
long *getRowMapA(long index);

void setMapA(long index, long data);
void setMapB(long index, long data);
void setModulus(long index, long data);
void setScalar(long index, long data);

long getMapA(long index);
long getMapB(long index);

void InitContiguousHostMapModulus(long phim, int n_rows);

void CudaEltwiseAddMod(long n_rows);
void CudaEltwiseAddMod(long n_rows, long scalar);
void CudaEltwiseSubMod(long n_rows);
void CudaEltwiseSubMod(long n_rows, long scalar);
void CudaEltwiseMultMod(long n_rows);
void CudaEltwiseMultMod(long n_rows, long scalar);

int cuda_add();
void init_gpu_ntt(unsigned int n);
void moveTwFtoGPU(unsigned long long gpu_powers_dev[], std::vector<unsigned long long>& gpu_powers, int k2, NTL::zz_pX& powers, unsigned long long gpu_powers_m_dev[]);
void gpu_mulMod(NTL::zz_pX& x, unsigned long long x_dev[], unsigned long long gpu_powers_m_dev[], unsigned long long p, int n);
void gpu_mulMod2(NTL::zz_pX& x, unsigned long long x_dev[], unsigned long long gpu_powers_m_dev[], unsigned long long p, int n);

void gpu_ntt(unsigned int n, NTL::zz_pX& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv);
void gpu_ntt(unsigned long long res[], unsigned int n, const NTL::zz_pX& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse);
void gpu_ntt(unsigned long long res[], unsigned int n, unsigned long long x[], unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse);
void gpu_ntt(NTL::vec_zz_p& res, unsigned int n, const NTL::zz_pX& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse);
void gpu_ntt(NTL::vec_zz_p& res, unsigned int n, const NTL::vec_zz_p& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse);

void gpu_ntt_forward(unsigned long long res[], unsigned int n, const NTL::zz_pX& x, unsigned long long q, const std::vector<unsigned long long>& gpu_powers, unsigned long long psi, unsigned long long psiinv);
void gpu_ntt_backward(NTL::vec_zz_p& res, unsigned int n, const NTL::vec_zz_p& x, unsigned long long q, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv);
void gpu_fused_polymul(NTL::vec_zz_p& res, unsigned long long a_dev[], const unsigned long long b_dev[], int n, unsigned long long x_dev[], unsigned long long q, const std::vector<unsigned long long>& gpu_powers, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv, int l, unsigned long long gpu_powers_dev[], unsigned long long gpu_ipowers_dev[]);
void gpu_ntt_forward_old(NTL::vec_zz_p& res, unsigned int n, const NTL::zz_pX& x, unsigned long long q, const std::vector<unsigned long long>& gpu_powers, unsigned long long psi, unsigned long long psiinv);
void gpu_ntt_backward_old(NTL::vec_zz_p& res, unsigned int n, const NTL::vec_zz_p& x, unsigned long long q, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv);
void gpu_addMod(unsigned long long x_dev[], long n, long dx, unsigned long long p);

#endif