#ifndef GPU_ACCEL_H
#define GPU_ACCEL_H

#include <helib/NumbTh.h>
#define THREADS_PER_BLOCK 1024


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
void gpu_ntt(unsigned int n, NTL::zz_pX& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv);
void gpu_ntt(unsigned long long res[], unsigned int n, const NTL::zz_pX& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse);
void gpu_ntt(unsigned long long res[], unsigned int n, unsigned long long x[], unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse);
void gpu_ntt(NTL::vec_zz_p& res, unsigned int n, const NTL::zz_pX& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse);
void gpu_ntt(NTL::vec_zz_p& res, unsigned int n, const NTL::vec_zz_p& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse);

void gpu_ntt_forward(NTL::vec_zz_p& res, unsigned int n, const NTL::zz_pX& x, unsigned long long q, const std::vector<unsigned long long>& gpu_powers, unsigned long long psi, unsigned long long psiinv);
void gpu_ntt_backward(NTL::vec_zz_p& res, unsigned int n, const NTL::vec_zz_p& x, unsigned long long q, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv);

#endif