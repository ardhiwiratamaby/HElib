#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpu_accel.cuh"
#include <NTL/ZZVec.h>
// #include <cub/cub.cuh>

__global__ void KernelMulMod(unsigned long long a[], const unsigned long long b[], unsigned long long q){
  int global_tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

  __extension__ unsigned __int128 temp_storage = b[global_tid];
  temp_storage *= a[global_tid];
  a[global_tid] = temp_storage % q;
}

__global__ void KernelMulMod(unsigned long long a[], const unsigned long long b[], unsigned long long q, int n){
  int global_tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

  if(global_tid<n){
    __extension__ unsigned __int128 temp_storage = b[global_tid];
    temp_storage *= a[global_tid];
    a[global_tid] = temp_storage % q;
  }
}

// __device__ __forceinline__ void singleBarrett(unsigned long long& a, unsigned& q, unsigned& mu, int& qbit)  // ??
__device__ __forceinline__ void singleBarrett(unsigned __int128& a, unsigned long long& q, unsigned long long& mu, int& qbit)  // ??
{  
    // unsigned long long rx;
    unsigned __int128 rx;
    rx = a >> (qbit - 2);
    rx *= mu;
    rx >>= qbit + 2;
    rx *= q;
    a -= rx;

    a -= q * (a >= q);
}

#if 1 //CT & GS for 2048 ntt points
__global__ void CTBasedNTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[], int n_of_groups)
{
    // unsigned long long *a;
    // a = &d_a[blockIdx.x * 2048];
    int local_tid = threadIdx.x;

    extern __shared__ unsigned long long shared_array[];

    //Ardhi: To my understanding, this is just load coeffs into shared memory, one thread pickup two coeffs to shmem
    //for ex: shared_array[tid]=a[tid]
    #pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying to shared memory from global
        int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * 2048]; //Ardhi: if blockIdx.x is always zero, why this is needed??
    }

    #pragma unroll
    for (int length = 1; length < 2048; length *= 2)
    {  // iterations for ntt
        int step = (2048 / length) / 2;
        int psi_step = local_tid / step;
        int target_index = psi_step * step * 2 + local_tid % step;

        psi_step = (local_tid + blockIdx.x * 1024) / step;

        // unsigned long long psi = psi_powers[length + psi_step];

        unsigned long long psi = psi_powers[n_of_groups + (threadIdx.x + blockIdx.x * THREADS_PER_BLOCK)/(1024/length)];

        unsigned long long first_target_value = shared_array[target_index];
        // unsigned long long temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow
        __extension__ unsigned __int128 temp_storage = shared_array[target_index + step];  // this is for eliminating the possibility of overflow

        temp_storage *= psi;

        // singleBarrett(temp_storage, q, mu, qbit);
        temp_storage %= q;
        unsigned long long second_target_value = temp_storage;

        unsigned long long target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);

        shared_array[target_index] = target_result;

        first_target_value += q * (first_target_value < second_target_value);

        shared_array[target_index + step] = first_target_value - second_target_value;

        n_of_groups *= 2;

        __syncthreads();
    }

    #pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying back to global from shared
        int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * 2048] = shared_array[global_tid];
    }
}

__global__ void GSBasedINTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[], int n, int n_of_groups)
{
    int local_tid = threadIdx.x;

    extern __shared__ unsigned long long shared_array[];

    // unsigned long q2 = (q + 1) >> 1;

    #pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying to shared memory from global
        int global_tid = local_tid + iteration_num * 1024;
        shared_array[global_tid] = a[global_tid + blockIdx.x * 2048];
    }

    __syncthreads();


    int group_size = 2;// =n/n_of_groups
    n_of_groups = n/2;
    #pragma unroll
    for (int length = 1024; length >= 1; length /= 2)
    {  // iterations for intt
        int step = (2048 / length) / 2;

        int psi_step = local_tid / step;
        int target_index = psi_step * step * 2 + local_tid % step;

        psi_step = (local_tid + blockIdx.x * 1024) / step;

        // unsigned long long psiinv = psiinv_powers[length + psi_step];
        int n_of_thread_per_group = (group_size/2);
        unsigned long long psiinv = psiinv_powers[n_of_groups + (threadIdx.x + blockIdx.x * THREADS_PER_BLOCK)/n_of_thread_per_group];
        //unsigned long long psiinv = psiinv_powers[n_of_groups + global_tid/n_of_thread_per_group];
        // printf("n_of_groups: %d, global_tid %d, target_index: %d, psi_index: %d\n", length, (threadIdx.x + blockIdx.x * THREADS_PER_BLOCK), target_index, length + (threadIdx.x + blockIdx.x * THREADS_PER_BLOCK)/n_of_thread_per_group);

        unsigned long long first_target_value = shared_array[target_index];
        unsigned long long second_target_value = shared_array[target_index + step];

        unsigned long long target_result = first_target_value + second_target_value;

        target_result -= q * (target_result >= q);

        // shared_array[target_index] = (target_result >> 1) + q2 * (target_result & 1);
        shared_array[target_index] =target_result;

        first_target_value += q * (first_target_value < second_target_value);

        // unsigned long long temp_storage = first_target_value - second_target_value;
        __extension__ unsigned __int128 temp_storage = first_target_value - second_target_value;

        temp_storage *= psiinv;

        // singleBarrett(temp_storage, q, mu, qbit);
        temp_storage %= q;
        
        unsigned long long temp_storage_low = temp_storage;

        // shared_array[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
        shared_array[target_index + step] = temp_storage_low;
        
        n_of_groups /= 2;
        group_size *= 2;
        
        __syncthreads();
    }

    #pragma unroll
    for (int iteration_num = 0; iteration_num < 2; iteration_num++)
    {  // copying back to global from shared
        int global_tid = local_tid + iteration_num * 1024;
        a[global_tid + blockIdx.x * 2048] = shared_array[global_tid];
    }
    
}
#endif

__global__ void CTBasedNTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[], int n, int n_of_groups)
{
    int global_tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

    int group_size = n/n_of_groups;
    int n_of_thread_per_group = group_size/2; //we assume that 1 thread manages 2 coefficients
    int step = n_of_thread_per_group;

    int target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid % n_of_thread_per_group);

    unsigned long long psi = psi_powers[n_of_groups + global_tid/n_of_thread_per_group];

    unsigned long long first_target_value = a[target_index];

    unsigned __int128 temp_storage = a[target_index + step];  // this is for eliminating the possibility of overflow

    temp_storage *= psi;

    // singleBarrett(temp_storage, q, mu, qbit);
    temp_storage %= q;
    unsigned long long second_target_value = temp_storage;

    unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    a[target_index + step] = first_target_value - second_target_value;

}

__global__ void GSBasedINTTInnerSingle(unsigned long long a[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psiinv_powers[], int n, unsigned long long n_inv, int n_of_groups)
{
    int global_tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

    // unsigned long q2 = (q + 1) >> 1;

    int group_size = n/n_of_groups;
    int n_of_thread_per_group = group_size/2; //we assume that 1 thread manages 2 coefficients
    int step = n_of_thread_per_group;

    int target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid % n_of_thread_per_group);

    unsigned long long psiinv = psiinv_powers[n_of_groups + global_tid/n_of_thread_per_group];

    // printf("n_of_groups: %d, global_tid %d, target_index: %d, psi_index: %d\n", n_of_groups, global_tid, target_index, n_of_groups + global_tid/n_of_thread_per_group);

    unsigned long long first_target_value = a[target_index];

    unsigned long long second_target_value = a[target_index + step];

    unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    // a[target_index] = (target_result >> 1) + q2 * (target_result & 1);
    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    unsigned __int128 temp_storage = first_target_value - second_target_value;

    temp_storage *= psiinv;

    // singleBarrett(temp_storage, q, mu, qbit);
    temp_storage %= q;

    unsigned long long temp_storage_low = temp_storage;

    // a[target_index + step] = (temp_storage_low >> 1) + q2 * (temp_storage_low & 1);
    a[target_index + step] = temp_storage_low;

    if(n_of_groups == 1){
    //Ardhi: below code for normalization of the result i.e a[i]*(1/n)
        group_size = n/1;
        n_of_thread_per_group = group_size/2; //we assume that 1 thread manages 2 coefficients
        step = n_of_thread_per_group;

        target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid % n_of_thread_per_group);

        // unsigned __int128 temp_storage;
        
        temp_storage = a[target_index];
        temp_storage *= n_inv;
        // singleBarrett(temp_storage, q, mu, qbit);
        temp_storage %= q;
        a[target_index] = temp_storage;

        temp_storage = a[target_index+step];
        temp_storage *= n_inv;
        // singleBarrett(temp_storage, q, mu, qbit);
        temp_storage %= q;
        a[target_index+step] = temp_storage;
    }
        
}

__global__ void CTBasedNTTInnerSingleAndMulMod(unsigned long long a[], unsigned long long b[], unsigned long long q, unsigned long long mu, int qbit, unsigned long long psi_powers[], int n, int n_of_groups)
{
    int global_tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;

    int group_size = n/n_of_groups;
    int n_of_thread_per_group = group_size/2; //we assume that 1 thread manages 2 coefficients
    int step = n_of_thread_per_group;

    int target_index = (global_tid/n_of_thread_per_group)*group_size+(global_tid % n_of_thread_per_group);

    unsigned long long psi = psi_powers[n_of_groups + global_tid/n_of_thread_per_group];

    unsigned long long first_target_value = a[target_index];

    unsigned __int128 temp_storage = a[target_index + step];  // this is for eliminating the possibility of overflow

    temp_storage *= psi;

    // singleBarrett(temp_storage, q, mu, qbit);
    temp_storage %= q;
    unsigned long long second_target_value = temp_storage;

    unsigned long long target_result = first_target_value + second_target_value;

    target_result -= q * (target_result >= q);

    a[target_index] = target_result;

    first_target_value += q * (first_target_value < second_target_value);

    a[target_index + step] = first_target_value - second_target_value;

    if(n_of_groups == n/2){

    }
}

#include <stdlib.h>
#include <random>

unsigned long long modpow64(unsigned long long a, unsigned long long b, unsigned long long mod)  // calculates (<a> ** <b>) mod <mod>
{
    unsigned long long res = 1;

    if (1 & b)
        res = a;

    while (b != 0)
    {
        b = b >> 1;
        // unsigned long long t64 = (unsigned long long)a * a;
        // a = t64 % mod;
        __extension__ unsigned __int128 t128 = (unsigned __int128)a * a;
        a = t128 % mod;

        if (b & 1)
        {
            // unsigned long long r64 = (unsigned long long)a * res;
            // res = r64 % mod;
            __extension__ unsigned __int128 r128 = (unsigned __int128)a * res;
            res = r128 % mod;
        }

    }
    return res;
}

unsigned long long bitReverse(unsigned long long a, int bit_length)  // reverses the bits for twiddle factor calculation
{
    // cout<<"a: "<<a;
    unsigned long long res = 0;

    for (int i = 0; i < bit_length; i++)
    {
        res <<= 1;
        res = (a & 1) | res;
        a >>= 1;
    }

    // cout<<"\n reverse A: "<<res<<endl;
    return res;
}

std::random_device dev;  // uniformly distributed integer random number generator that produces non-deterministic random numbers
std::mt19937_64 rng(dev());  // pseudo-random generator of 64 bits with a state size of 19937 bits

void randomArray64(unsigned long long a[], int n, unsigned long long q)
{
    std::uniform_int_distribution<unsigned long long> randnum(0, q - 1);  // uniformly distributed random integers on the closed interval [a, b] according to discrete probability

    for (int i = 0; i < n; i++)
    {
        a[i] = randnum(rng);
    }
}

void fillTablePsi64(unsigned long long psi, unsigned long long q, unsigned long long psiinv, unsigned long long psiTable[], unsigned long long psiinvTable[], unsigned int n)  // twiddle factors computation
{
    for (unsigned int i = 0; i < n; i++)
    {
        psiTable[i] = modpow64(psi, bitReverse(i, log2(n)), q);
        // cout<<"\npsi: "<<psi<<" bitRev: "<<bitReverse(i, log2(n))<<" mod: "<<q<<" psi^bitRev mod q: "<<psiTable[i];
        psiinvTable[i] = modpow64(psiinv, bitReverse(i, log2(n)), q);
    }
}

//device variable;
#define magicNumber 3
// long *d_A, *d_B, *d_C, *d_modulus, *d_scalar;
// size_t bytes;
// long d_phim, d_n_rows;

// long *contiguousHostMapA, *contiguousHostMapB, *contiguousModulus, *scalarPerRow;

void InitContiguousHostMapModulus(long phim, int n_rows, long *contiguousHostMapA, long *contiguousHostMapB, long *contiguousModulus, long *scalarPerRow){
  // contiguousHostMapA = (long *)malloc(magicNumber*phim*n_rows*sizeof(long));
  // contiguousHostMapB = (long *)malloc(magicNumber*phim*n_rows*sizeof(long));
  // contiguousModulus = (long *)malloc(magicNumber*n_rows*sizeof(long));
  // scalarPerRow = (long *)malloc(magicNumber*n_rows*sizeof(long));

  cudaMallocHost(&contiguousHostMapA, magicNumber*phim*n_rows*sizeof(long));
  cudaMallocHost(&contiguousHostMapB, magicNumber*phim*n_rows*sizeof(long));
  cudaMallocHost(&contiguousModulus, magicNumber*n_rows*sizeof(long));
  cudaMallocHost(&scalarPerRow, magicNumber*n_rows*sizeof(long));

}

void setScalar(long index, long data, long *scalarPerRow){
  scalarPerRow[index] = data;
}

void setMapA(long index, long data, long *contiguousHostMapA){
  contiguousHostMapA[index] = data;
}

void setMapB(long index, long data, long *contiguousHostMapB){
  contiguousHostMapB[index] = data;
}

void setRowMapA(long offset, long *source, long *contiguousHostMapA, long d_phim)
{
  memcpy(contiguousHostMapA+offset, source, d_phim*sizeof(long));
}

void setRowMapB(long offset, const long *source, long *contiguousHostMapB, long d_phim)
{
  memcpy(contiguousHostMapB+offset, source, d_phim*sizeof(long));
}

long *getRowMapB(long index, long *contiguousHostMapB){
  return &contiguousHostMapB[index];
}

long *getRowMapA(long index, long *contiguousHostMapA){
  return &contiguousHostMapA[index];
}

long getMapA(long index, long *contiguousHostMapA){
  return contiguousHostMapA[index];
}

long getMapB(long index, long *contiguousHostMapB){
  return contiguousHostMapB[index];
}

void setModulus(long index, long data, long *contiguousModulus, long n_rows){
  if(contiguousModulus == NULL) //Ardhi: I don't know why this may happen
    contiguousModulus = (long *)malloc(magicNumber*n_rows*sizeof(long));
  // try
  // {
  //   long check = *contiguousModulus;
  // }
  // catch(const std::exception& e)
  // {
  //   cudaMallocHost(&contiguousModulus, magicNumber*n_rows*sizeof(long));
  // }
  
  contiguousModulus[index] = data;
}

void InitGPUBuffer(long phim, int n_rows, long *d_A, long *d_B, long *d_C, long *d_modulus, long *d_scalar, int bytes){
  // d_phim = phim;
  // d_n_rows = n_rows;

  bytes = magicNumber*phim*n_rows*sizeof(long);

  // Allocate memory for arrays d_A, d_B, and d_C on device
  CHECK(cudaMalloc(&d_A, bytes));
  CHECK(cudaMalloc(&d_B, bytes));
  CHECK(cudaMalloc(&d_C, bytes));
  CHECK(cudaMalloc(&d_modulus, magicNumber*n_rows*sizeof(long)));
  CHECK(cudaMalloc(&d_scalar, magicNumber*n_rows*sizeof(long)));
}

void DestroyGPUBuffer(){
    // Free GPU memory

    // cudaFree(d_C);
    // cudaFree(d_A);
    // cudaFree(d_B);

    // cudaFree(d_modulus);
    // cudaFree(d_scalar);
}


__global__ void kernel_addMod(long *a, long *b, long *result, long size, long *d_modulus, long phim){
    // printf("==kernel code== Hi there, this is Ardhi\n");
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
      result[tid] = a[tid] + b[tid];
      // result[tid] %= modulus;
      if(result[tid] >= d_modulus[tid/phim])
        result[tid] -= d_modulus[tid/phim];
    }
}

#define debug_impl 0

void CudaEltwiseAddMod(long actual_nrows, long *contiguousHostMapA, long *contiguousHostMapB, long *contiguousModulus, long *d_A, long *d_B, long *d_C, long *d_modulus, int bytes, long d_phim){
    // Copy data from host arrays A and B to device arrays d_A and d_B
    CHECK(cudaMemcpy(d_A, contiguousHostMapA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, contiguousHostMapB, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_modulus, contiguousModulus, actual_nrows*sizeof(long), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(d_phim*actual_nrows) / thr_per_blk );

    // Launch kernel
    kernel_addMod<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, d_phim*actual_nrows, d_modulus, d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    CHECK(cudaMemcpy(contiguousHostMapA, d_C, bytes, cudaMemcpyDeviceToHost));


  // //allocate data in device memory
  // thrust::device_vector<long> d_result(size); //I assume all should be zeros
  // // thrust::device_vector<long> d_a(a,a + size);
  // // thrust::device_vector<long> d_b(b,b + size);

  // thrust::device_vector<long> d_a(size);
  // thrust::device_vector<long> d_b(size);

  // thrust::host_vector<long> h_a(a, a+size);
  // thrust::host_vector<long> h_b(b, b+size);

  // d_a = h_a;
  // d_b = h_b;

  // thrust::host_vector<long> clone_h_a(size);
  // thrust::host_vector<long> clone_h_b(size);
  // thrust::host_vector<long> clone_result(size);

  // thrust::copy(d_a.begin(), d_a.end(), clone_h_a.begin());
  // thrust::copy(d_b.begin(), d_b.end(), clone_h_b.begin());

  // for(long j=0; j<size; j++){
  //   if(h_a[j] != clone_h_a[j] || h_b[j] != clone_h_b[j])
  //     std::cout<<"Data missmatch detected\n";
  // }


  // // thrust::device_vector<long> d_a(size, 1);
  // // thrust::device_vector<long> d_b(size,5);

  // thrust::device_vector<long> d_modululus(size);

  // //fill d_modulus with modulus
  // thrust::fill(d_modululus.begin(), d_modululus.end(), modulus);

  // //result = a + b
  // thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_result.begin(), thrust::plus<long>());

  // //result = result mod modulus
  // thrust::transform(d_result.begin(), d_result.end(), d_modululus.begin(), d_result.begin(), thrust::modulus<long>());

  // //copy the result back to CPU
  // thrust::copy(d_result.begin(), d_result.end(), clone_result.begin());

  // for(long j=0; j<size; j++){
  //   *result = clone_result[j];
  //   ++result;
  // }


}

__global__ void kernel_addModScalar(long *a, long *scalar, long *result, long size, long *d_modulus, long phim){
    // printf("==kernel code== Hi there, this is Ardhi\n");
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
      result[tid] = a[tid] + scalar[tid/phim];
      // result[tid] %= modulus;
      if(result[tid] >= d_modulus[tid/phim])
        result[tid] -= d_modulus[tid/phim];
    }
}

void CudaEltwiseAddMod(long actual_nrows, long scalar, long *contiguousHostMapA, long *scalarPerRow, long *contiguousModulus, long *d_A, long *d_C, long *d_scalar, long *d_modulus, int bytes, long d_phim){
    // Copy data from host arrays A and B to device arrays d_A and d_B
    CHECK(cudaMemcpy(d_A, contiguousHostMapA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scalar, scalarPerRow, actual_nrows*sizeof(long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_modulus, contiguousModulus, actual_nrows*sizeof(long), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(d_phim*actual_nrows) / thr_per_blk );

    // Launch kernel
    kernel_addModScalar<<< blk_in_grid, thr_per_blk >>>(d_A, d_scalar, d_C, d_phim*actual_nrows, d_modulus, d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    CHECK(cudaMemcpy(contiguousHostMapA, d_C, bytes, cudaMemcpyDeviceToHost));

#if 0
  //allocate data in device memory
  thrust::device_vector<long> d_result(result,result + size);
  thrust::device_vector<long> d_a(a,a + size);
  thrust::device_vector<long> d_b(size);
  thrust::device_vector<long> d_modululus(size);

  //fill b with scalar
  thrust::fill(d_b.begin(), d_b.end(), scalar);

  //fill d_modulus with modulus
  thrust::fill(d_modululus.begin(), d_modululus.end(), modulus);

  //result = a + b
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_result.begin(), thrust::plus<long>());

  //result = result mod modulus
  thrust::transform(d_result.begin(), d_result.end(), d_modululus.begin(), d_result.begin(), thrust::modulus<long>());

  //copy the result back to CPU
  thrust::copy(d_result.begin(), d_result.end(), result);
#endif
}

__global__ void kernel_subMod(long *a, long *b, long *result, long size, long *d_modulus, long phim){
    // printf("==kernel code== Hi there, this is Ardhi\n");
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
      if (a[tid] >= b[tid]) {
        result[tid] = a[tid] - b[tid];
      } else {
        result[tid] = a[tid] + d_modulus[tid/phim] - b[tid];
      }
    }
}

void CudaEltwiseSubMod(long actual_nrows, long *contiguousHostMapA, long *contiguousHostMapB, long *contiguousModulus, long *d_A, long *d_B, long *d_C, long *d_modulus, int bytes, long d_phim){
    // Copy data from host arrays A and B to device arrays d_A and d_B
    CHECK(cudaMemcpy(d_A, contiguousHostMapA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, contiguousHostMapB, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_modulus, contiguousModulus, actual_nrows*sizeof(long), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(d_phim*actual_nrows) / thr_per_blk );

    // Launch kernel
    kernel_subMod<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, d_phim*actual_nrows, d_modulus, d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    CHECK(cudaMemcpy(contiguousHostMapA, d_C, bytes, cudaMemcpyDeviceToHost));

#if 0
  //allocate data in device memory
  thrust::device_vector<long> d_result(result,result + size);
  thrust::device_vector<long> d_a(a,a + size);
  thrust::device_vector<long> d_b(b,b + size);
  thrust::device_vector<long> d_modululus(size);

  //fill d_modulus with modulus
  thrust::fill(d_modululus.begin(), d_modululus.end(), modulus);

  //b = -b
  thrust::transform(d_b.begin(), d_b.end(), d_b.begin(), thrust::negate<long>());

  //result = a + b
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_result.begin(), thrust::plus<long>());

  //result = result mod modulus
  thrust::transform(d_result.begin(), d_result.end(), d_modululus.begin(), d_result.begin(), thrust::modulus<long>());

  //copy the result back to CPU
  thrust::copy(d_result.begin(), d_result.end(), result);
#endif
}

__global__ void kernel_subModScalar(long *a, long *scalar, long *result, long size, long *d_modulus, long phim){
    // printf("==kernel code== Hi there, this is Ardhi\n");
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < size){
      if (a[tid] >= scalar[tid/phim]) {
        result[tid] = a[tid] - scalar[tid/phim];
      } else {
        result[tid] = a[tid] + d_modulus[tid/phim] - scalar[tid/phim];
      }
    }
}

void CudaEltwiseSubMod(long actual_nrows, long scalar, long *contiguousHostMapA, long *scalarPerRow, long *contiguousModulus, long *d_A, long *d_C, long *d_scalar, long *d_modulus, int bytes, long d_phim){
    // Copy data from host arrays A and B to device arrays d_A and d_B
    CHECK(cudaMemcpy(d_A, contiguousHostMapA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scalar, scalarPerRow, actual_nrows*sizeof(long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_modulus, contiguousModulus, actual_nrows*sizeof(long), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(d_phim*actual_nrows) / thr_per_blk );

    // Launch kernel
    kernel_subModScalar<<< blk_in_grid, thr_per_blk >>>(d_A, d_scalar, d_C, d_phim*actual_nrows, d_modulus, d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    CHECK(cudaMemcpy(contiguousHostMapA, d_C, bytes, cudaMemcpyDeviceToHost));

#if 0
  //allocate data in device memory
  thrust::device_vector<long> d_result(result,result + size);
  thrust::device_vector<long> d_a(a,a + size);
  thrust::device_vector<long> d_b(size);
  thrust::device_vector<long> d_modululus(size);

  //negate scalar
  scalar = -scalar;

  //fill b with -scalar
  thrust::fill(d_b.begin(), d_b.end(), scalar);

  //fill d_modulus with modulus
  thrust::fill(d_modululus.begin(), d_modululus.end(), modulus);

  //result = a + b
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_result.begin(), thrust::plus<long>());

  //result = result mod modulus
  thrust::transform(d_result.begin(), d_result.end(), d_modululus.begin(), d_result.begin(), thrust::modulus<long>());

  //copy the result back to CPU
  thrust::copy(d_result.begin(), d_result.end(), result);
#endif
}

// inline uint64_t DivideUInt128UInt64Lo(uint64_t x1, uint64_t x0, uint64_t y) {
//   uint128_t n =
//       (static_cast<uint128_t>(x1) << 64) | (static_cast<uint128_t>(x0));
//   uint128_t q = n / y;

//   return static_cast<uint64_t>(q);
// }

__device__ __int128 myMod(__int128 x, long m) {
    return (x%m + m)%m;
}



__device__ long mul_mod(long a, long b, long m) {
    if (!((a | b) & (0xFFFFFFFFULL << 32))) return a * b % m;

    long d = 0, mp2 = m >> 1;
    int i;
    if (a >= m) a %= m;
    if (b >= m) b %= m;
    for (i = 0; i < 64; ++i) {
        d = (d > mp2) ? (d << 1) - m : d << 1;
        if (a & 0x8000000000000000ULL) d += b;
        if (d >= m) d -= m;
        a <<= 1;
    }
    return d;
}

__device__ __int128 myMod3(__int128 a,long b)
{
    if(a < 0)
      a *= -1;
    
    return b - (a%b);
}

long sp_CorrectDeficit(long a, long n)
{
   return a >= 0 ? a : a+n;
}

long sp_CorrectExcess(long a, long n)
{
   return a-n >= 0 ? a-n : a;
}

// long MulMod(long a, long b, long n, double ninv) {
//   long q = long( double(a) * (double(b) * ninv) );
//   unsigned long rr = cast_unsigned(a)*cast_unsigned(b) - cast_unsigned(q)*cast_unsigned(n);
//   long r = sp_CorrectDeficit(rr, n);
//   return sp_CorrectExcess(r, n);
// }



std::ostream&
operator<<( std::ostream& dest, __int128_t value )
{
    std::ostream::sentry s( dest );
    if ( s ) {
        __uint128_t tmp = value < 0 ? -value : value;
        char buffer[ 128 ];
        char* d = std::end( buffer );
        do
        {
            -- d;
            *d = "0123456789"[ tmp % 10 ];
            tmp /= 10;
        } while ( tmp != 0 );
        if ( value < 0 ) {
            -- d;
            *d = '-';
        }
        int len = std::end( buffer ) - d;
        if ( dest.rdbuf()->sputn( d, len ) != len ) {
            dest.setstate( std::ios_base::badbit );
        }
    }
    return dest;
}

// Returns most-significant bit of the input
inline uint64_t MSB(uint64_t input) {
  return static_cast<uint64_t>(std::log2l(input));
}

inline uint64_t Log2(uint64_t x) { return MSB(x); }

template <int InputModFactor>
uint64_t ReduceMod(uint64_t x, uint64_t modulus,
                   const uint64_t* twice_modulus = nullptr,
                   const uint64_t* four_times_modulus = nullptr) {

  if (InputModFactor == 1) {
    return x;
  }
  if (InputModFactor == 2) {
    if (x >= modulus) {
      x -= modulus;
    }
    return x;
  }
  if (InputModFactor == 4) {
    if (x >= *twice_modulus) {
      x -= *twice_modulus;
    }
    if (x >= modulus) {
      x -= modulus;
    }
    return x;
  }
  if (InputModFactor == 8) {

    if (x >= *four_times_modulus) {
      x -= *four_times_modulus;
    }
    if (x >= *twice_modulus) {
      x -= *twice_modulus;
    }
    if (x >= modulus) {
      x -= modulus;
    }
    return x;
  }

  return x;
}

__extension__ typedef __int128 int128_t;
__extension__ typedef unsigned __int128 uint128_t;

// Returns low 64bit of 128b/64b where x1=high 64b, x0=low 64b
inline uint64_t DivideUInt128UInt64Lo(uint64_t x1, uint64_t x0, uint64_t y) {
  uint128_t n =
      (static_cast<uint128_t>(x1) << 64) | (static_cast<uint128_t>(x0));
  uint128_t q = n / y;

  return static_cast<uint64_t>(q);
}

/// @brief Pre-computes a Barrett factor with which modular multiplication can
/// be performed more efficiently
class MultiplyFactor {
 public:
  MultiplyFactor() = default;

  /// @brief Computes and stores the Barrett factor floor((operand << bit_shift)
  /// / modulus). This is useful when modular multiplication of the form
  /// (x * operand) mod modulus is performed with same modulus and operand
  /// several times. Note, passing operand=1 can be used to pre-compute a
  /// Barrett factor for multiplications of the form (x * y) mod modulus, where
  /// only the modulus is re-used across calls to modular multiplication.
  MultiplyFactor(uint64_t operand, uint64_t bit_shift, uint64_t modulus)
      : m_operand(operand) {
    uint64_t op_hi = operand >> (64 - bit_shift);
    uint64_t op_lo = (bit_shift == 64) ? 0 : (operand << bit_shift);

    m_barrett_factor = DivideUInt128UInt64Lo(op_hi, op_lo, modulus);
  }

  /// @brief Returns the pre-computed Barrett factor
  inline uint64_t BarrettFactor() const { return m_barrett_factor; }

  /// @brief Returns the operand corresponding to the Barrett factor
  inline uint64_t Operand() const { return m_operand; }

 private:
  uint64_t m_operand;
  uint64_t m_barrett_factor;
};

inline uint128_t MultiplyUInt64(uint64_t x, uint64_t y) {
  return uint128_t(x) * uint128_t(y);
}

inline void MultiplyUInt64(uint64_t x, uint64_t y, uint64_t* prod_hi,
                           uint64_t* prod_lo) {
  uint128_t prod = MultiplyUInt64(x, y);
  *prod_hi = static_cast<uint64_t>(prod >> 64);
  *prod_lo = static_cast<uint64_t>(prod);
}

__device__ __int128 flooredDivision(__int128 a, long b)
{
    if(a/b > 0)
      return a/b;

    if(a%b == 0)
      return (a/b);
    else
      return (a/b)-1;

}
__device__ __int128 myMod2(__int128 a,long b)
{
    return a - b * flooredDivision(a, b);
}

__global__ void kernel_mulMod(long *a, long *b, long *result, long size, long *d_modulus, long phim){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid < size){
    // __int128 temp_res=0;
    // __int128 temp_a=0;
    // __int128 temp_b=0;
    __int128 temp_storage= a[tid];
    // temp_a = a[tid];
    // temp_b = b[tid];

    // temp_res = temp_a * temp_b;
    temp_storage *= b[tid];
    // temp_res = temp_res%modulus;
    // temp_res = myMod2(temp_res, d_modulus[tid/phim]);

    // d_result[tid] = temp_res;
    // result[tid] %= modulus;
    result[tid]=temp_storage % d_modulus[tid/phim];

    // result[tid]=mul_mod(a[tid], b[tid], modulus);
  }
}

void CudaEltwiseMultMod(long actual_nrows, long *contiguousHostMapA, long *contiguousHostMapB, long *contiguousModulus, long *d_A, long *d_B, long *d_C, long *d_modulus, int bytes, long d_phim){
	// HELIB_NTIMER_START(CudaEltwiseMultMod_CudaMemCpyHD);
    // Copy data from host arrays A and B to device arrays d_A and d_B
    CHECK(cudaMemcpy(d_A, contiguousHostMapA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, contiguousHostMapB, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_modulus, contiguousModulus, actual_nrows*sizeof(long), cudaMemcpyHostToDevice));
	// HELIB_NTIMER_STOP(CudaEltwiseMultMod_CudaMemCpyHD);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(d_phim*actual_nrows) / thr_per_blk );
	// HELIB_NTIMER_START(CudaEltwiseMultMod_kernel);
    // Launch kernel
    kernel_mulMod<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, d_phim*actual_nrows, d_modulus, d_phim);
    // cudaDeviceSynchronize();
	// HELIB_NTIMER_STOP(CudaEltwiseMultMod_kernel);

	// HELIB_NTIMER_START(CudaEltwiseMultMod_CudaMemCpyDH);
    // Copy data from device array d_C to host array result
    CHECK(cudaMemcpy(contiguousHostMapA, d_C, bytes, cudaMemcpyDeviceToHost));
	// HELIB_NTIMER_STOP(CudaEltwiseMultMod_CudaMemCpyDH);

#if 0
//HEXL Naive MulMod
  constexpr int64_t beta = -2;
  constexpr int64_t alpha = 62;  // ensures alpha - beta = 64
  int const InputModFactor=1;

  uint64_t gamma = Log2(InputModFactor);
  // HEXL_UNUSED(gamma);

  const uint64_t ceil_log_mod = Log2(modulus) + 1;  // "n" from Algorithm 2
  uint64_t prod_right_shift = ceil_log_mod + beta;

  // Barrett factor "mu"
  // TODO(fboemer): Allow MultiplyFactor to take bit shifts != 64
  uint64_t barr_lo =
      MultiplyFactor(uint64_t(1) << (ceil_log_mod + alpha - 64), 64, modulus)
          .BarrettFactor();

  const uint64_t twice_modulus = 2 * modulus;

  for (size_t i = 0; i < size; ++i) {
    uint64_t prod_hi, prod_lo, c2_hi, c2_lo, Z;

    uint64_t x = ReduceMod<InputModFactor>(*a, modulus, &twice_modulus);
    uint64_t y = ReduceMod<InputModFactor>(*b, modulus, &twice_modulus);

    // Multiply inputs
    MultiplyUInt64(x, y, &prod_hi, &prod_lo);

    // floor(U / 2^{n + beta})
    uint64_t c1 = (prod_lo >> (prod_right_shift)) +
                  (prod_hi << (64 - (prod_right_shift)));

    // c2 = floor(U / 2^{n + beta}) * mu
    MultiplyUInt64(c1, barr_lo, &c2_hi, &c2_lo);

    // alpha - beta == 64, so we only need high 64 bits
    uint64_t q_hat = c2_hi;

    // only compute low bits, since we know high bits will be 0
    Z = prod_lo - q_hat * modulus;

    // Conditional subtraction
    *result = (Z >= modulus) ? (Z - modulus) : Z;

    if(NTL::MulMod(a[i], b[i], modulus) != *result)
        std::cout<<"MulMod Missmatch Detected. j="<<i<<"/"<<size<<" CPU: "<<NTL::MulMod(a[i], b[i], modulus)<<" GPU: "<<*result<<" A: "<<a[i]<<" B: "<<b[i]<<" Mod: "<<modulus<<std::endl;
    else 
        std::cout<<"MulMod Matched. j="<<i<<"/"<<size<<" CPU: "<<NTL::MulMod(a[i], b[i], modulus)<<" GPU: "<<*result<<" A: "<<a[i]<<" B: "<<b[i]<<" Mod: "<<modulus<<std::endl;

    ++a;
    ++b;
    ++result;
  }
  #endif

//Cuda thrust naive
  // //allocate data in device memory
  // thrust::device_vector<long> d_result(result,result + size);
  // thrust::device_vector<long> d_a(a,a + size);
  // thrust::device_vector<long> d_b(b,b + size);
  // thrust::device_vector<long> d_modululus(size);

  // //fill d_modulus with modulus
  // thrust::fill(d_modululus.begin(), d_modululus.end(), modulus);

  // //result = a * b
  // thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_result.begin(), thrust::multiplies<long>());

  // //result = result mod modulus
  // thrust::transform(d_result.begin(), d_result.end(), d_modululus.begin(), d_result.begin(), thrust::modulus<long>());

  // //copy the result back to CPU
  // thrust::copy(d_result.begin(), d_result.end(), result);
}

__global__ void kernel_mulModScalar(long *a, long *scalar, long *result, long size, long *d_modulus, long phim){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if(tid < size){
    __int128 temp_res=0;
    __int128 temp_a=0;
    __int128 temp_b=0;

    temp_a = a[tid];
    temp_b = scalar[tid/phim];

    temp_res = temp_a * temp_b;
    // temp_res = temp_res%modulus;
    temp_res = myMod2(temp_res, d_modulus[tid/phim]);

    // d_result[tid] = temp_res;
    // result[tid] %= modulus;
    result[tid]=temp_res;

    // result[tid]=mul_mod(a[tid], b[tid], modulus);
  }
}

void CudaEltwiseMultMod(long actual_nrows, long scalar, long *contiguousHostMapA, long *scalarPerRow, long *contiguousModulus, long *d_A, long *d_C, long *d_scalar, long *d_modulus, int bytes, long d_phim){
   // Copy data from host arrays A and B to device arrays d_A and d_B
    CHECK(cudaMemcpy(d_A, contiguousHostMapA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scalar, scalarPerRow, actual_nrows*sizeof(long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_modulus, contiguousModulus, actual_nrows*sizeof(long), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(d_phim*actual_nrows) / thr_per_blk );

    // Launch kernel
    kernel_mulModScalar<<< blk_in_grid, thr_per_blk >>>(d_A, d_scalar, d_C, d_phim*actual_nrows, d_modulus, d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    CHECK(cudaMemcpy(contiguousHostMapA, d_C, bytes, cudaMemcpyDeviceToHost));

#if 0
  //allocate data in device memory
  thrust::device_vector<long> d_result(result,result + size);
  thrust::device_vector<long> d_a(a,a + size);
  thrust::device_vector<long> d_b(size);
  thrust::device_vector<long> d_modululus(size);

  //fill b with scalar
  thrust::fill(d_b.begin(), d_b.end(), scalar);

  //fill d_modulus with modulus
  thrust::fill(d_modululus.begin(), d_modululus.end(), modulus);

  //result = a * b
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_result.begin(), thrust::multiplies<long>());

  //result = result mod modulus
  thrust::transform(d_result.begin(), d_result.end(), d_modululus.begin(), d_result.begin(), thrust::modulus<long>());

  //copy the result back to CPU
  thrust::copy(d_result.begin(), d_result.end(), result);
#endif
}


int cuda_add() {
  std::cout<<"Inside cu_add\n";

  std::cout<<"Run cuda thrust\n";
  // test<<<1,1>>>();
  // Generate random data serially.
  thrust::default_random_engine rng(1337);
  thrust::uniform_real_distribution<double> dist(-50.0, 50.0);
  thrust::host_vector<double> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

  // Transfer to device and compute the sum.
  thrust::device_vector<double> d_vec = h_vec;
  double x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

  return 0;
}

void gpu_ntt(unsigned int n, NTL::zz_pX& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv){
    int size_array = sizeof(unsigned long long) * n;
    int size = sizeof(unsigned long long);
    bool check=true;
//    unsigned q = 536608769, psi = 284166, psiinv = 208001377;  // parameter initialization
    // unsigned long long q = 536608769, psi = 206332156, psiinv = 416707834;  // parameter initialization
    // unsigned long long q = 280349076803813377, psi = 33780496399778535, psiinv = 141007519160814319;  // parameter initialization
    unsigned int q_bit = ceil(std::log2(q));
    // unsigned int q_bit = 29;

    /****************************************************************
    BEGIN
    cudamalloc, memcpy, etc... for gpu
    */

    unsigned long long* psiTable = (unsigned long long*)malloc(size_array);
    unsigned long long* psiinvTable = (unsigned long long*)malloc(size_array);
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, n); //gel psi psi

    //copy powers of psi and psi inverse tables to device
    unsigned long long* psi_powers, * psiinv_powers;

    cudaMalloc(&psi_powers, size_array);
    cudaMalloc(&psiinv_powers, size_array);

    cudaMemcpy(psi_powers, psiTable, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(psiinv_powers, psiinvTable, size_array, cudaMemcpyHostToDevice);

    // we print these because we forgot them every time :)
    std::cout << "n = " << n << std::endl;
    std::cout << "q = " << q << std::endl;
    std::cout << "Psi = " << psi << std::endl;
    std::cout << "Psi Inverse = " << psiinv << std::endl;

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    // double mu1 = powl(2, 2 * bit_length);
    // unsigned mu = mu1 / q;
    // unsigned long long mu = (__float128)2^(2*bit_length) / q;
    unsigned __int128 mu1 = 1;
    mu1 = mu1 << (2*bit_length);
    unsigned long long mu = mu1/q;

    // unsigned long long mu = 289881905946523498;


    unsigned long long* a;
    cudaMallocHost(&a, sizeof(unsigned long long) * n);
    // randomArray64(a, n, q); //fill array with random numbers between 0 and q - 1
    cudaMemset(a, 0, size_t(n)*sizeof(unsigned long long));
    long dx = deg(x);
    for(int i=0; i <= dx; i++)
      a[i] = NTL::rep(x.rep[i]);

    unsigned long long* res_a;
    cudaMallocHost(&res_a, sizeof(unsigned long long) * n);

    unsigned long long* d_a;
    cudaMalloc(&d_a, size_array);

    cudaMemcpyAsync(d_a, a, size_array, cudaMemcpyHostToDevice, 0);

    /*
    END
    cudamalloc, memcpy, etc... for gpu
    ****************************************************************/

    
    /****************************************************************
    BEGIN
    Kernel Calls
    */
    // CTBasedNTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned long long), 0>>>(d_a, q, mu, bit_length, psi_powers);
    // GSBasedINTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned long long), 0>>>(d_a, q, mu, bit_length, psiinv_powers);
    long n_inv = NTL::InvMod(n, q);
    int num_blocks = n/(THREADS_PER_BLOCK*2);
    #pragma unroll
    for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
    { 
        CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psi_powers, n, n_of_groups);
    }
    #pragma unroll
    for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2)
    {
        GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psiinv_powers, n, n_inv, n_of_groups);
    }
    /*
    END
    Kernel Calls
    ****************************************************************/

    cudaMemcpyAsync(res_a, d_a, size_array, cudaMemcpyDeviceToHost, 0);  // do this in async 

    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

    bool correct = 1;
    if (check) //check the correctness of results
    {
        for (int i = 0; i < n; i++)
        {
            std::cout<<"a: "<<a[i]<<" res_a: "<<res_a[i]<<std::endl;
            if (a[i] != res_a[i])
            {
                correct = 0;
                break;
            }
        }
    }

    if (correct)
        std::cout << "\nNTT and INTT are working correctly." << std::endl;
    else
        std::cout << "\nNTT and INTT are not working correctly." << std::endl;

    cudaFreeHost(a); cudaFreeHost(res_a);  
    cudaFree(d_a);
}

//device buffers
unsigned long long* psi_powers, * psiinv_powers;
unsigned long long* a;
unsigned long long* d_a;
unsigned long long* d_b;
unsigned long long* psiTable;
unsigned long long* psiinvTable;
// cudaStream_t stream[32];

void init_gpu_ntt(unsigned int n){
    int size_array = sizeof(unsigned long long) * n;
    cudaMalloc(&psi_powers, size_array);
    cudaMalloc(&psiinv_powers, size_array);
    cudaMallocHost(&a, sizeof(unsigned long long) * n);
    cudaMalloc(&d_a, size_array);
    cudaMalloc(&d_b, size_array);
    psiTable = (unsigned long long*)malloc(size_array);
    psiinvTable = (unsigned long long*)malloc(size_array);

    // for (int i = 0; i < 32; ++i)
    //   cudaStreamCreate(&stream[i]); 
}

void moveTwFtoGPU(unsigned long long gpu_powers_dev[], std::vector<unsigned long long>& gpu_powers, int k2, NTL::zz_pX& powers, unsigned long long gpu_powers_m_dev[], long zMStar_dev[], long zMStar_h[], long mm, long target_dev[], long target_h[]){
    CHECK(cudaMemcpy(gpu_powers_dev, gpu_powers.data(), k2 * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_powers_m_dev, powers.rep.data(), powers.rep.length() * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(zMStar_dev, zMStar_h, mm*sizeof(long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(target_dev, target_h, mm*sizeof(long), cudaMemcpyHostToDevice));
}

void gpu_ntt(unsigned long long res[], unsigned int n, const NTL::zz_pX& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse){
    int size_array = sizeof(unsigned long long) * n;
    int size = sizeof(unsigned long long);
    bool check=true;
//    unsigned q = 536608769, psi = 284166, psiinv = 208001377;  // parameter initialization
    // unsigned long long q = 536608769, psi = 206332156, psiinv = 416707834;  // parameter initialization
    // unsigned long long q = 280349076803813377, psi = 33780496399778535, psiinv = 141007519160814319;  // parameter initialization
    unsigned int q_bit = ceil(std::log2(q));
    // unsigned int q_bit = 29;

    /****************************************************************
    BEGIN
    cudamalloc, memcpy, etc... for gpu
    */

    unsigned long long* psiTable = (unsigned long long*)malloc(size_array);
    unsigned long long* psiinvTable = (unsigned long long*)malloc(size_array);
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, n); //gel psi psi

    //copy powers of psi and psi inverse tables to device

    // cudaMalloc(&psi_powers, size_array);
    // cudaMalloc(&psiinv_powers, size_array);

    cudaMemcpy(psi_powers, psiTable, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(psiinv_powers, psiinvTable, size_array, cudaMemcpyHostToDevice);

    // we print these because we forgot them every time :)
    // std::cout << "n = " << n << std::endl;
    // std::cout << "q = " << q << std::endl;
    // std::cout << "Psi = " << psi << std::endl;
    // std::cout << "Psi Inverse = " << psiinv << std::endl;

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    // double mu1 = powl(2, 2 * bit_length);
    // unsigned mu = mu1 / q;
    // unsigned long long mu = (__float128)2^(2*bit_length) / q;
    unsigned __int128 mu1 = 1;
    mu1 = mu1 << (2*bit_length);
    unsigned long long mu = mu1/q;

    // unsigned long long mu = 289881905946523498;



    // randomArray64(a, n, q); //fill array with random numbers between 0 and q - 1
    cudaMemset(a, 0, size_t(n)*sizeof(unsigned long long));
    long dx = deg(x);
    for(int i=0; i <= dx; i++)
      a[i] = NTL::rep(x.rep[i]);

    // unsigned long long* res_a;
    // cudaMallocHost(&res_a, sizeof(unsigned long long) * n);



    cudaMemcpyAsync(d_a, a, size_array, cudaMemcpyHostToDevice, 0);

    /*
    END
    cudamalloc, memcpy, etc... for gpu
    ****************************************************************/

    
    /****************************************************************
    BEGIN
    Kernel Calls
    */
    // if(!inverse)
    //   CTBasedNTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned long long), 0>>>(d_a, q, mu, bit_length, psi_powers);
    // else
    //   GSBasedINTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned long long), 0>>>(d_a, q, mu, bit_length, psiinv_powers);
    long n_inv = NTL::InvMod(n, q);
    int num_blocks = n/(THREADS_PER_BLOCK*2);
    if(!inverse){
      #pragma unroll
      for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
      { 
          CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psi_powers, n, n_of_groups);
      }
    }
    else {
      #pragma unroll
      for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2)
      {
          GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psiinv_powers, n, n_inv, n_of_groups);
      }
    }
    
    /*
    END
    Kernel Calls
    ****************************************************************/

    cudaMemcpyAsync(a, d_a, size_array, cudaMemcpyDeviceToHost, 0);  // do this in async 

    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

    for(long i=0; i<n; i++)
      res[i] = a[i];

    // cudaFreeHost(a);
    // cudaFree(d_a);
}

void gpu_ntt(NTL::vec_zz_p& res, unsigned int n, const NTL::zz_pX& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse){
    int size_array = sizeof(unsigned long long) * n;
    int size = sizeof(unsigned long long);
    bool check=true;
//    unsigned q = 536608769, psi = 284166, psiinv = 208001377;  // parameter initialization
    // unsigned long long q = 536608769, psi = 206332156, psiinv = 416707834;  // parameter initialization
    // unsigned long long q = 280349076803813377, psi = 33780496399778535, psiinv = 141007519160814319;  // parameter initialization
    unsigned int q_bit = ceil(std::log2(q));
    // unsigned int q_bit = 29;

    /****************************************************************
    BEGIN
    cudamalloc, memcpy, etc... for gpu
    */

    unsigned long long* psiTable = (unsigned long long*)malloc(size_array);
    unsigned long long* psiinvTable = (unsigned long long*)malloc(size_array);
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, n); //gel psi psi

    //copy powers of psi and psi inverse tables to device

    // cudaMalloc(&psi_powers, size_array);
    // cudaMalloc(&psiinv_powers, size_array);

    cudaMemcpy(psi_powers, psiTable, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(psiinv_powers, psiinvTable, size_array, cudaMemcpyHostToDevice);

    // we print these because we forgot them every time :)
    // std::cout << "n = " << n << std::endl;
    // std::cout << "q = " << q << std::endl;
    // std::cout << "Psi = " << psi << std::endl;
    // std::cout << "Psi Inverse = " << psiinv << std::endl;

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    // double mu1 = powl(2, 2 * bit_length);
    // unsigned mu = mu1 / q;
    // unsigned long long mu = (__float128)2^(2*bit_length) / q;
    unsigned __int128 mu1 = 1;
    mu1 = mu1 << (2*bit_length);
    unsigned long long mu = mu1/q;

    // unsigned long long mu = 289881905946523498;



    // randomArray64(a, n, q); //fill array with random numbers between 0 and q - 1

    // cudaMemset(a, 0, size_t(n)*sizeof(unsigned long long));
    // long dx = deg(x);
    // for(int i=0; i <= dx; i++)
    //   a[i] = NTL::rep(x.rep[i]);

    long dx = deg(x);
    for(int i=0; i < n; i++)
      if(i<=dx)
        a[i] = NTL::rep(x.rep[i]);
      else
        a[i]=0;

    // unsigned long long* res_a;
    // cudaMallocHost(&res_a, sizeof(unsigned long long) * n);



    cudaMemcpy(d_a, a, size_array, cudaMemcpyHostToDevice);

    /*
    END
    cudamalloc, memcpy, etc... for gpu
    ****************************************************************/

    
    /****************************************************************
    BEGIN
    Kernel Calls
    */
    // if(!inverse)
    //   CTBasedNTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned long long), 0>>>(d_a, q, mu, bit_length, psi_powers);
    // else
    //   GSBasedINTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned long long), 0>>>(d_a, q, mu, bit_length, psiinv_powers);
    long n_inv = NTL::InvMod(n, q);
    int num_blocks = n/(THREADS_PER_BLOCK*2);
    if(!inverse){
      #pragma unroll
      for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
      { 
          CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psi_powers, n, n_of_groups);
      }
    }
    else {
      #pragma unroll
      for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2)
      {
          GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psiinv_powers, n, n_inv, n_of_groups);
      }
    }
    /*
    END
    Kernel Calls
    ****************************************************************/
    #if 0 //Ardhi: check for correctness//flipped from the above codes//this code is not used
      unsigned long long* a_clone;
      cudaMallocHost(&a_clone, sizeof(unsigned long long) * n);
      cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

      if(inverse){
        #pragma unroll
        for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
        { 
            CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psi_powers, n, n_of_groups);
        }
      }
      else {
        #pragma unroll
        for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2)
        {
            GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psiinv_powers, n, n_inv, n_of_groups);
        }
      }
      cudaMemcpy(a_clone, d_a, size_array, cudaMemcpyDeviceToHost);  // do this in async 
      for (long i = 0; i < n; i++)
      {
        if(a_clone[i] != a[i]){
              printf("a: %lu, a_clone: %lu, x: %lu\n", a[i], a_clone[i], rep(x.rep[i]));
              throw std::runtime_error("a and a_clone missmatch");
        }
      }

      if(!inverse){
        #pragma unroll
        for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
        { 
            CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psi_powers, n, n_of_groups);
        }
      }
      else {
        #pragma unroll
        for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2)
        {
            GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psiinv_powers, n, n_inv, n_of_groups);
        }
      }
    #endif

    cudaMemcpy(a, d_a, size_array, cudaMemcpyDeviceToHost);  // do this in async 

    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

    for(long i=0; i<n; i++)
      res[i] = a[i];


    // cudaFreeHost(a);
    // cudaFree(d_a);
}

void gpu_ntt(unsigned long long res[], unsigned int n, unsigned long long x[], unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse){
    int size_array = sizeof(unsigned long long) * n;
    int size = sizeof(unsigned long long);
    bool check=true;
//    unsigned q = 536608769, psi = 284166, psiinv = 208001377;  // parameter initialization
    // unsigned long long q = 536608769, psi = 206332156, psiinv = 416707834;  // parameter initialization
    // unsigned long long q = 280349076803813377, psi = 33780496399778535, psiinv = 141007519160814319;  // parameter initialization
    unsigned int q_bit = ceil(std::log2(q));
    // unsigned int q_bit = 29;

    /****************************************************************
    BEGIN
    cudamalloc, memcpy, etc... for gpu
    */

    // unsigned long long* psiTable = (unsigned long long*)malloc(size_array);
    // unsigned long long* psiinvTable = (unsigned long long*)malloc(size_array);
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, n); //gel psi psi

    //copy powers of psi and psi inverse tables to device
    // unsigned long long* psi_powers, * psiinv_powers;

    // cudaMalloc(&psi_powers, size_array);
    // cudaMalloc(&psiinv_powers, size_array);

    cudaMemcpy(psi_powers, psiTable, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(psiinv_powers, psiinvTable, size_array, cudaMemcpyHostToDevice);

    // we print these because we forgot them every time :)
    // std::cout << "n = " << n << std::endl;
    // std::cout << "q = " << q << std::endl;
    // std::cout << "Psi = " << psi << std::endl;
    // std::cout << "Psi Inverse = " << psiinv << std::endl;

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    // double mu1 = powl(2, 2 * bit_length);
    // unsigned mu = mu1 / q;
    // unsigned long long mu = (__float128)2^(2*bit_length) / q;
    unsigned __int128 mu1 = 1;
    mu1 = mu1 << (2*bit_length);
    unsigned long long mu = mu1/q;

    // unsigned long long mu = 289881905946523498;


    // unsigned long long* a;
    // cudaMallocHost(&a, sizeof(unsigned long long) * n);
    // randomArray64(a, n, q); //fill array with random numbers between 0 and q - 1

    cudaMemset(a, 0, size_t(n)*sizeof(unsigned long long));
    for(int i=0; i < n; i++)
      a[i] = x[i];

    // unsigned long long* res_a;
    // cudaMallocHost(&res_a, sizeof(unsigned long long) * n);

    // unsigned long long* d_a;
    // cudaMalloc(&d_a, size_array);

    cudaMemcpyAsync(d_a, a, size_array, cudaMemcpyHostToDevice, 0);

    /*
    END
    cudamalloc, memcpy, etc... for gpu
    ****************************************************************/

    
    /****************************************************************
    BEGIN
    Kernel Calls
    */
    // if(!inverse)
    //   CTBasedNTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned long long), 0>>>(d_a, q, mu, bit_length, psi_powers);
    // else
    //   GSBasedINTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned long long), 0>>>(d_a, q, mu, bit_length, psiinv_powers);
    long n_inv = NTL::InvMod(n, q);
    int num_blocks = n/(THREADS_PER_BLOCK*2);
    if(!inverse){
      #pragma unroll
      for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
      { 
          CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psi_powers, n, n_of_groups);
      }
    }
    else {
      #pragma unroll
      for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2)
      {
          GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psiinv_powers, n, n_inv, n_of_groups);
      }
    }
    /*
    END
    Kernel Calls
    ****************************************************************/

    cudaMemcpyAsync(a, d_a, size_array, cudaMemcpyDeviceToHost, 0);  // do this in async 

    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

    for(long i=0; i<n; i++)
      res[i] = a[i];

    // cudaFreeHost(a);
    // cudaFree(d_a);
}

void gpu_ntt(NTL::vec_zz_p& res, unsigned int n, const NTL::vec_zz_p& x, unsigned long long q, unsigned long long psi, unsigned long long psiinv, bool inverse){
    int size_array = sizeof(unsigned long long) * n;
    int size = sizeof(unsigned long long);
    bool check=true;
//    unsigned q = 536608769, psi = 284166, psiinv = 208001377;  // parameter initialization
    // unsigned long long q = 536608769, psi = 206332156, psiinv = 416707834;  // parameter initialization
    // unsigned long long q = 280349076803813377, psi = 33780496399778535, psiinv = 141007519160814319;  // parameter initialization
    unsigned int q_bit = ceil(std::log2(q));
    // unsigned int q_bit = 29;

    /****************************************************************
    BEGIN
    cudamalloc, memcpy, etc... for gpu
    */

    // unsigned long long* psiTable = (unsigned long long*)malloc(size_array);
    // unsigned long long* psiinvTable = (unsigned long long*)malloc(size_array);
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, n); //gel psi psi

    //copy powers of psi and psi inverse tables to device
    // unsigned long long* psi_powers, * psiinv_powers;

    // cudaMalloc(&psi_powers, size_array);
    // cudaMalloc(&psiinv_powers, size_array);

    cudaMemcpy(psi_powers, psiTable, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(psiinv_powers, psiinvTable, size_array, cudaMemcpyHostToDevice);

    // we print these because we forgot them every time :)
    // std::cout << "n = " << n << std::endl;
    // std::cout << "q = " << q << std::endl;
    // std::cout << "Psi = " << psi << std::endl;
    // std::cout << "Psi Inverse = " << psiinv << std::endl;

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    // double mu1 = powl(2, 2 * bit_length);
    // unsigned mu = mu1 / q;
    // unsigned long long mu = (__float128)2^(2*bit_length) / q;
    unsigned __int128 mu1 = 1;
    mu1 = mu1 << (2*bit_length);
    unsigned long long mu = mu1/q;

    // unsigned long long mu = 289881905946523498;


    // unsigned long long* a;
    // cudaMallocHost(&a, sizeof(unsigned long long) * n);
    // randomArray64(a, n, q); //fill array with random numbers between 0 and q - 1

    // cudaMemset(a, 0, size_t(n)*sizeof(unsigned long long)); //Ardhi: this is not needed since all a's element is overrided
    for(int i=0; i < n; i++)
      a[i] = rep(x[i]);

    // unsigned long long* res_a;
    // cudaMallocHost(&res_a, sizeof(unsigned long long) * n);

    // unsigned long long* d_a;
    // cudaMalloc(&d_a, size_array);

    cudaMemcpy(d_a, a, size_array, cudaMemcpyHostToDevice);

    /*
    END
    cudamalloc, memcpy, etc... for gpu
    ****************************************************************/

    
    /****************************************************************
    BEGIN
    Kernel Calls
    */
    // if(!inverse)
    //   CTBasedNTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned long long), 0>>>(d_a, q, mu, bit_length, psi_powers);
    // else
    //   GSBasedINTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned long long), 0>>>(d_a, q, mu, bit_length, psiinv_powers);
    long n_inv = NTL::InvMod(n, q);
    int num_blocks = n/(THREADS_PER_BLOCK*2);
    if(!inverse){
      #pragma unroll
      for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
      { 
          CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psi_powers, n, n_of_groups);
      }
    }
    else {
      #pragma unroll
      for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2)
      {
          GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psiinv_powers, n, n_inv, n_of_groups);
      }
    }
    /*
    END
    Kernel Calls
    ****************************************************************/

    cudaMemcpy(a, d_a, size_array, cudaMemcpyDeviceToHost);  // do this in async 

    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

    for(long i=0; i<n; i++)
      res[i] = a[i];

    // cudaFreeHost(a);
    // cudaFree(d_a);
}

void gpu_ntt_forward(unsigned long long res[], unsigned int n, const NTL::zz_pX& x, unsigned long long q, const std::vector<unsigned long long>& gpu_powers, unsigned long long psi, unsigned long long psiinv){
    int size_array = sizeof(unsigned long long) * n;
    unsigned int q_bit = ceil(std::log2(q));

    /****************************************************************
    BEGIN
    cudamalloc, memcpy, etc... for gpu
    */

#if 0 //Ardhi: check psi computation
    unsigned long long* psiTable = (unsigned long long*)malloc(size_array);
    unsigned long long* psiinvTable = (unsigned long long*)malloc(size_array);
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, n); //gel psi psi

    for(long i=0; i<n; i++){
      if(psiTable[i] != gpu_powers[i]){
          throw std::runtime_error("psiTable and gpu_powers missmatch");
      }
    }
#endif

	HELIB_NTIMER_START(CudaMemCpyHD);

    // const unsigned long long *twiddle_factors = gpu_powers.data();
    cudaMemcpy(psi_powers, gpu_powers.data(), size_array, cudaMemcpyHostToDevice);

    // we print these because we forgot them every time :)
    // std::cout << "n = " << n << std::endl;
    // std::cout << "q = " << q << std::endl;
    // std::cout << "Psi = " << psi << std::endl;
    // std::cout << "Psi Inverse = " << psiinv << std::endl;

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    // double mu1 = powl(2, 2 * bit_length);
    // unsigned mu = mu1 / q;
    unsigned __int128 mu1 = 1;
    mu1 = mu1 << (2*bit_length);
    unsigned long long mu = mu1/q;

    long dx = deg(x);
    for(int i=0; i < n; i++)
      if(i<=dx)
        a[i] = NTL::rep(x.rep[i]);
      else
        a[i]=0;

    cudaMemcpy(res, a, size_array, cudaMemcpyHostToDevice);

	HELIB_NTIMER_STOP(CudaMemCpyHD);
    /*
    END
    cudamalloc, memcpy, etc... for gpu
    ****************************************************************/

    
    /****************************************************************
    BEGIN
    Kernel Calls
    */
	HELIB_NTIMER_START(KernelNTT);
    long n_inv = NTL::InvMod(n, q);
    int num_blocks = n/(THREADS_PER_BLOCK*2);
    int n_of_groups=1;
    #pragma unroll
    // for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
    for (n_of_groups = 1; (n/n_of_groups>2048); n_of_groups *= 2) //Ardhi: do the ntt until the working size is 2048 elements
    {
        CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(res, q, mu, bit_length, psi_powers, n, n_of_groups);
    }

    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

    // for(int i=0; i<32; i++){
    //     int offset = i*2048;
    //     CTBasedNTTInnerSingle<<<1, 1024, 2048 * sizeof(unsigned long long), stream[i]>>>(d_a+offset, q, mu, bit_length, psi_powers);
    // }

    CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 2048 * sizeof(unsigned long long), 0>>>(res, q, mu, bit_length, psi_powers, n_of_groups);
	HELIB_NTIMER_STOP(KernelNTT);

    /*
    END
    Kernel Calls
    ****************************************************************/
#if 0
	HELIB_NTIMER_START(CudaMemCpyDH);
    cudaMemcpy(a, d_a, size_array, cudaMemcpyDeviceToHost);  // do this in async 
	HELIB_NTIMER_STOP(CudaMemCpyDH);

    // cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

	HELIB_NTIMER_START(BufferCopy);
    for(long i=0; i<n; i++)
      res[i] = a[i]; //this is very expensive! around 8.5 seconds
	HELIB_NTIMER_STOP(BufferCopy);
#endif

#if 0
	HELIB_NTIMER_START(CudaMemCpyDH);
    cudaMemcpy(res.data(), d_a, size_array, cudaMemcpyDeviceToHost);  // do this in async 
	HELIB_NTIMER_STOP(CudaMemCpyDH);
#endif
}

void gpu_ntt_forward_old(NTL::vec_zz_p& res, unsigned int n, const NTL::zz_pX& x, unsigned long long q, const std::vector<unsigned long long>& gpu_powers, unsigned long long psi, unsigned long long psiinv){
    int size_array = sizeof(unsigned long long) * n;
    unsigned int q_bit = ceil(std::log2(q));

    /****************************************************************
    BEGIN
    cudamalloc, memcpy, etc... for gpu
    */

#if 0 //Ardhi: check psi computation
    unsigned long long* psiTable = (unsigned long long*)malloc(size_array);
    unsigned long long* psiinvTable = (unsigned long long*)malloc(size_array);
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, n); //gel psi psi

    for(long i=0; i<n; i++){
      if(psiTable[i] != gpu_powers[i]){
          throw std::runtime_error("psiTable and gpu_powers missmatch");
      }
    }

#endif

    // const unsigned long long *twiddle_factors = gpu_powers.data();
    cudaMemcpy(psi_powers, gpu_powers.data(), size_array, cudaMemcpyHostToDevice);

    // we print these because we forgot them every time :)
    // std::cout << "n = " << n << std::endl;
    // std::cout << "q = " << q << std::endl;
    // std::cout << "Psi = " << psi << std::endl;
    // std::cout << "Psi Inverse = " << psiinv << std::endl;

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    // double mu1 = powl(2, 2 * bit_length);
    // unsigned mu = mu1 / q;
    unsigned __int128 mu1 = 1;
    mu1 = mu1 << (2*bit_length);
    unsigned long long mu = mu1/q;

    long dx = deg(x);
    for(int i=0; i < n; i++)
      if(i<=dx)
        a[i] = NTL::rep(x.rep[i]);
      else
        a[i]=0;

    cudaMemcpy(d_a, a, size_array, cudaMemcpyHostToDevice);

    /*
    END
    cudamalloc, memcpy, etc... for gpu
    ****************************************************************/

    
    /****************************************************************
    BEGIN
    Kernel Calls
    */

    long n_inv = NTL::InvMod(n, q);
    int num_blocks = n/(THREADS_PER_BLOCK*2);
    #pragma unroll
    for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
    { 
        CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psi_powers, n, n_of_groups);
    }
    /*
    END
    Kernel Calls
    ****************************************************************/

    cudaMemcpy(a, d_a, size_array, cudaMemcpyDeviceToHost);  // do this in async 

    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

    for(long i=0; i<n; i++)
      res[i] = a[i];

}

void gpu_ntt_backward(NTL::vec_zz_p& res, unsigned int n, const NTL::vec_zz_p& x, unsigned long long q, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv){
    int size_array = sizeof(unsigned long long) * n;
    unsigned int q_bit = ceil(std::log2(q));

    /****************************************************************
    BEGIN
    cudamalloc, memcpy, etc... for gpu
    */

#if 0 //Ardhi: check psi computation
    unsigned long long* psiTable = (unsigned long long*)malloc(size_array);
    unsigned long long* psiinvTable = (unsigned long long*)malloc(size_array);
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, n); //gel psi psi

    for(long i=0; i<n; i++){
      if(psiinvTable[i] != gpu_ipowers[i]){
          throw std::runtime_error("psiTable and gpu_powers missmatch");
      }
    }

#endif

    // const unsigned long long *twiddle_factors = gpu_powers.data();
    cudaMemcpy(psiinv_powers, gpu_ipowers.data(), size_array, cudaMemcpyHostToDevice);

    // we print these because we forgot them every time :)
    // std::cout << "n = " << n << std::endl;
    // std::cout << "q = " << q << std::endl;
    // std::cout << "Psi = " << psi << std::endl;
    // std::cout << "Psi Inverse = " << psiinv << std::endl;

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    // double mu1 = powl(2, 2 * bit_length);
    // unsigned mu = mu1 / q;
    unsigned __int128 mu1 = 1;
    mu1 = mu1 << (2*bit_length);
    unsigned long long mu = mu1/q;

    for(int i=0; i < n; i++)
      a[i] = rep(x[i]);

    cudaMemcpy(d_a, a, size_array, cudaMemcpyHostToDevice);

    /*
    END
    cudamalloc, memcpy, etc... for gpu
    ****************************************************************/

    
    /****************************************************************
    BEGIN
    Kernel Calls
    */

    long n_inv = NTL::InvMod(n, q);
    int num_blocks = n/(THREADS_PER_BLOCK*2);

#if 1
    GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 2048 * sizeof(unsigned long long)>>>(d_a, q, mu, bit_length, psiinv_powers, n, n/2);

    if(n>2048){
      #pragma unroll
      // for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2) 
      for (int n_of_groups = (n/2048)/2; n_of_groups >= 1; n_of_groups /= 2) 
      {
          GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psiinv_powers, n, n_inv, n_of_groups);
      }
    }
#else
      #pragma unroll
      for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2) 
      {
          GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psiinv_powers, n, n_inv, n_of_groups);
      }
#endif
    /*
    END
    Kernel Calls
    ****************************************************************/

    cudaMemcpy(a, d_a, size_array, cudaMemcpyDeviceToHost);  // do this in async 

    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

    for(long i=0; i<n; i++)
      res[i] = a[i];

}

void gpu_ntt_backward_old(NTL::vec_zz_p& res, unsigned int n, const NTL::vec_zz_p& x, unsigned long long q, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv){
    int size_array = sizeof(unsigned long long) * n;
    unsigned int q_bit = ceil(std::log2(q));

    /****************************************************************
    BEGIN
    cudamalloc, memcpy, etc... for gpu
    */

#if 0 //Ardhi: check psi computation
    unsigned long long* psiTable = (unsigned long long*)malloc(size_array);
    unsigned long long* psiinvTable = (unsigned long long*)malloc(size_array);
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, n); //gel psi psi

    for(long i=0; i<n; i++){
      if(psiinvTable[i] != gpu_ipowers[i]){
          throw std::runtime_error("psiTable and gpu_powers missmatch");
      }
    }

#endif

    // const unsigned long long *twiddle_factors = gpu_powers.data();
    cudaMemcpy(psiinv_powers, gpu_ipowers.data(), size_array, cudaMemcpyHostToDevice);

    // we print these because we forgot them every time :)
    // std::cout << "n = " << n << std::endl;
    // std::cout << "q = " << q << std::endl;
    // std::cout << "Psi = " << psi << std::endl;
    // std::cout << "Psi Inverse = " << psiinv << std::endl;

    //generate parameters for barrett
    unsigned int bit_length = q_bit;
    // double mu1 = powl(2, 2 * bit_length);
    // unsigned mu = mu1 / q;
    unsigned __int128 mu1 = 1;
    mu1 = mu1 << (2*bit_length);
    unsigned long long mu = mu1/q;

    for(int i=0; i < n; i++)
      a[i] = rep(x[i]);

    cudaMemcpy(d_a, a, size_array, cudaMemcpyHostToDevice);

    /*
    END
    cudamalloc, memcpy, etc... for gpu
    ****************************************************************/

    
    /****************************************************************
    BEGIN
    Kernel Calls
    */

    long n_inv = NTL::InvMod(n, q);
    int num_blocks = n/(THREADS_PER_BLOCK*2);
    #pragma unroll
    for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2)
    {
        GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(d_a, q, mu, bit_length, psiinv_powers, n, n_inv, n_of_groups);
    }
    /*
    END
    Kernel Calls
    ****************************************************************/

    cudaMemcpy(a, d_a, size_array, cudaMemcpyDeviceToHost);  // do this in async 

    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

    for(long i=0; i<n; i++)
      res[i] = a[i];

}

__global__ void kernel_mulMod(long *a, long *b, long *result, long d_modulus, long phim){
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  __int128 temp_res=0;
  __int128 temp_a=0;
  __int128 temp_b=0;

  temp_a = a[tid];
  temp_b = b[tid];

  temp_res = temp_a * temp_b;
  temp_res = myMod2(temp_res, d_modulus);

  result[tid]=temp_res;
}

void gpu_fused_polymul(NTL::vec_zz_p& res, unsigned long long a_dev[], const unsigned long long b_dev[], int n, unsigned long long n_inv, unsigned long long x_dev[], unsigned long long q, 
const std::vector<unsigned long long>& gpu_powers, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv, int l, unsigned long long gpu_powers_dev[], unsigned long long gpu_ipowers_dev[], cudaStream_t stream){
  int size_array = sizeof(unsigned long long) * n;
  unsigned int q_bit = ceil(std::log2(q));

  //generate parameters for barrett
  unsigned int bit_length = q_bit;
  // double mu1 = powl(2, 2 * bit_length);
  // unsigned mu = mu1 / q;
  unsigned __int128 mu1 = 1;
  mu1 = mu1 << (2*bit_length);
  unsigned long long mu = mu1/q;


	HELIB_NTIMER_START(KernelNTT);
  // long n_inv = NTL::InvMod(n, q);
  int num_blocks = n/(THREADS_PER_BLOCK*2);
  int n_of_groups=1;
  #pragma unroll
  // for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
  for (n_of_groups = 1; (n/n_of_groups>2048); n_of_groups *= 2) //Ardhi: do the ntt until the working size is 2048 elements
  {
      CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(x_dev, q, mu, bit_length, gpu_powers_dev, n, n_of_groups);
  }

  // cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job

  CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 2048 * sizeof(unsigned long long), stream>>>(x_dev, q, mu, bit_length, gpu_powers_dev, n_of_groups);
	HELIB_NTIMER_STOP(KernelNTT);

	HELIB_NTIMER_START(KernelMulMod);
  KernelMulMod<<<num_blocks*2, THREADS_PER_BLOCK, 0, stream>>>(x_dev, b_dev, q); //a = a*b mod q
	HELIB_NTIMER_STOP(KernelMulMod);

	HELIB_NTIMER_START(KernelNTT_inv);
  GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 2048 * sizeof(unsigned long long), stream>>>(x_dev, q, mu, bit_length, gpu_ipowers_dev, n, n/2);

  if(n>2048){
    #pragma unroll
    // for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2) 
    for (int n_of_groups = (n/2048)/2; n_of_groups >= 1; n_of_groups /= 2) 
    {
        GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(x_dev, q, mu, bit_length, gpu_ipowers_dev, n, n_inv, n_of_groups);
    }
  }
  HELIB_NTIMER_STOP(KernelNTT_inv);

  // res.SetLength(l);
	// HELIB_NTIMER_START(CudaMemCpyDH);
    // cudaMemcpy(res.data(), x_dev, l*sizeof(unsigned long long), cudaMemcpyDeviceToHost);  // do this in async 
	// HELIB_NTIMER_STOP(CudaMemCpyDH);
}

void gpu_mulMod(NTL::zz_pX& x, unsigned long long x_dev[], unsigned long long gpu_powers_m_dev[], unsigned long long p, int n, cudaStream_t stream){
    long dx = deg(x);
    // memset(a,0, n*8);
    // memcpy(a, x.rep.data(), (dx+1)*sizeof(unsigned long long));
  HELIB_NTIMER_START(gpu_mulMod_cuMemSet);
    cudaMemsetAsync(x_dev, 0, n*8, stream);
  HELIB_NTIMER_STOP(gpu_mulMod_cuMemSet);

  HELIB_NTIMER_START(gpu_mulMod_cpyHD);
    CHECK(cudaMemcpyAsync(x_dev, x.rep.data(), (dx+1)*sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
  HELIB_NTIMER_STOP(gpu_mulMod_cpyHD);

  HELIB_NTIMER_START(gpu_mulMod_kernel);
    KernelMulMod<<<ceil(((double)dx+1)/THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(x_dev, gpu_powers_m_dev, p, dx+1);
  HELIB_NTIMER_STOP(gpu_mulMod_kernel);

    // CHECK(cudaMemcpy((void *)x.rep.data(), x_dev, (dx+1)*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
}

__global__ void add_mod_kernel(unsigned long long x_dev[], long n, long l, unsigned long long p) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i + n < l) {
        // x[i - n].LoopHole() = NTL::AddMod(rep(x[i - n]), rep(x[i]), p);

        x_dev[i] = x_dev[i] + x_dev[i+n];
        if(x_dev[i] >= p)
          x_dev[i] -= p;
    }
}

void gpu_addMod(unsigned long long x_dev[], long n, long dx, unsigned long long p, cudaStream_t stream){
      int n_blocks= ceil(((double)(dx+1) - n + 1) / 512);
      add_mod_kernel<<<n_blocks, 512, 0, stream>>>(x_dev, n, dx, p);
}

void gpu_mulMod2(NTL::zz_pX& x, unsigned long long x_dev[], unsigned long long x_pinned[], unsigned long long gpu_powers_m_dev[], unsigned long long p, int n, cudaStream_t stream){
    // long dx = deg(x);
    // memset(a,0, n*8);
    // memcpy(a, x.rep.data(), (dx+1)*sizeof(unsigned long long));
    // cudaMemset(x_dev, 0, n*8);
    // CHECK(cudaMemcpy(x_dev, x.rep.data(), (dx+1)*sizeof(unsigned long long), cudaMemcpyHostToDevice));
	HELIB_NTIMER_START(AfterPolyMul_mulMod_kernel);
    KernelMulMod<<<ceil((double)n/THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>>(x_dev, gpu_powers_m_dev, p, n);
	HELIB_NTIMER_STOP(AfterPolyMul_mulMod_kernel);


if(stream == 0){
#if 1 //Ardhi: when the iFFT is async we can disable this memory copy
  // x.SetLength(n);
	HELIB_NTIMER_START(AfterPolyMul_mulMod_cpyDH);
    CHECK(cudaMemcpy(x.rep.data(), x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  // CHECK(cudaMemcpy(x_pinned, x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    x.normalize();
	HELIB_NTIMER_STOP(AfterPolyMul_mulMod_cpyDH);
#endif
}
#if 0
	HELIB_NTIMER_START(AfterPolyMul_mulMod_buffer);
  memcpy(x.rep.data(), x_pinned, n*sizeof(unsigned long long));
	HELIB_NTIMER_STOP(AfterPolyMul_mulMod_buffer);
#endif
}

inline void checkCudaError(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        printf("%s\n", msg);
        printf("Error: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

__global__ void parallel_copy(long m, unsigned long long x_pinned[], unsigned long long x_dev[], long zMStar[], long target_dev[])
{
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    // if(i == 0)
    //   printf("zMStar[%d]: %ld\n", i, zMStar[i]);
    // if (i < m && zMStar[i] != 0){
    //   // long j=0;
    //   // #endif 0
    //     // if (zMStar[i] != 0){
    //     //     j = atomicAdd(&counter[0], 1);
    //     //     // j =  atomicExch(counter, counter[0]+1);
    //     //     // j++;

    //     //     printf("tid %d write %llu to j=%ld zMStar[%ld]: %ld j: %ld m: %ld\n",i,x_dev[i], j, i, zMStar[i], j, m);
    //     //     // printf("zMStar[%ld]: %ld ", i, zMStar[i]);
    //     //     x_pinned[j] = x_dev[i];
    //     // }
    //     // else
    //     //     printf("!!! zero zMStar[%ld]: %ld j: %ld\n", i, zMStar[i], j);
    //   // #endif

    //   unsigned long long data = x_dev[i];
    //   int sum=0;
    //   for(int k=0; k < i; k++)
    //     sum += zMStar[k];

    //   // printf("tid %d write %llu to x_pinned[%d] \n", i, x_dev[i], sum);
    //   x_pinned[sum] = data;
    // }

    // for (long i = 0, j = 0; i < m; i++){
    //   if (zMStar[i] != 0){
    //       x_pinned[j++] = x_dev[i];
    //   }
    // }

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < m && zMStar[i] != 0){

      x_pinned[target_dev[i]] = x_dev[i];
      
      // int *sum;
      // if(i<10000){
      //   for(int k=0; k < i; k++)
      //       sum += zMStar[k];
      // }else{
      //   // Determine temporary device storage requirements
      //   void     *d_temp_storage = NULL;
      //   size_t   temp_storage_bytes = 0;
      //   cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, zMStar, sum, i);
      //   // Allocate temporary storage
      //   cudaMalloc(&d_temp_storage, temp_storage_bytes);
      //   // Run sum-reduction
      //   cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, zMStar, sum, i);
      //   x_pinned[sum[0]] = x_dev[i]; 
      // }
    }
}

void gpu_parallel_copy(long m, unsigned long long *x_pinned, unsigned long long *x_dev, long *zMStar_gpu, NTL::vec_long& y_h, long target_dev[], cudaStream_t stream){
    int numBlocks = (m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // unsigned long long *counter;
    // CHECK(cudaMalloc(&counter,sizeof(unsigned long long)));
    // unsigned long long counter_h = 0;
    // CHECK(cudaMemcpy(counter, &counter_h, sizeof(long), cudaMemcpyHostToDevice));

    // CHECK(cudaMemset(counter, 0, size_t(1)*sizeof(unsigned long long)));
    parallel_copy<<<numBlocks, THREADS_PER_BLOCK, 0, stream>>>(m, x_pinned, x_dev, zMStar_gpu, target_dev);
    // parallel_copy<<<1,1>>>(m, x_pinned, x_dev, zMStar_gpu, counter);



    CHECK(cudaMemcpyAsync(y_h.data(), x_pinned, y_h.length()*sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
}

//Ardhi: this is for blocking version of the bluestein
#if 1

void gpu_fused_polymul(NTL::vec_zz_p& res, unsigned long long a_dev[], const unsigned long long b_dev[], int n, unsigned long long n_inv, unsigned long long x_dev[], unsigned long long q, 
const std::vector<unsigned long long>& gpu_powers, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long psi, unsigned long long psiinv, int l, unsigned long long gpu_powers_dev[], unsigned long long gpu_ipowers_dev[]){
  int size_array = sizeof(unsigned long long) * n;
  unsigned int q_bit = ceil(std::log2(q));

  //generate parameters for barrett
  unsigned int bit_length = q_bit;
  unsigned __int128 mu1 = 1;
  mu1 = mu1 << (2*bit_length);
  unsigned long long mu = mu1/q;


	HELIB_NTIMER_START(KernelNTT);
  int num_blocks = n/(THREADS_PER_BLOCK*2);
  int n_of_groups=1;
  #pragma unroll
  // for (int n_of_groups = 1; n_of_groups < n; n_of_groups *= 2)
  for (n_of_groups = 1; (n/n_of_groups>2048); n_of_groups *= 2) //Ardhi: do the ntt until the working size is 2048 elements
  {
      CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(x_dev, q, mu, bit_length, gpu_powers_dev, n, n_of_groups);
  }

  CTBasedNTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 2048 * sizeof(unsigned long long)>>>(x_dev, q, mu, bit_length, gpu_powers_dev, n_of_groups);
	HELIB_NTIMER_STOP(KernelNTT);

	HELIB_NTIMER_START(KernelMulMod);
  KernelMulMod<<<num_blocks*2, THREADS_PER_BLOCK>>>(x_dev, b_dev, q); //a = a*b mod q
	HELIB_NTIMER_STOP(KernelMulMod);

	HELIB_NTIMER_START(KernelNTT_inv);
  GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK, 2048 * sizeof(unsigned long long)>>>(x_dev, q, mu, bit_length, gpu_ipowers_dev, n, n/2);

  if(n>2048){
    #pragma unroll
    // for (int n_of_groups = n/2; n_of_groups >= 1; n_of_groups /= 2) 
    for (int n_of_groups = (n/2048)/2; n_of_groups >= 1; n_of_groups /= 2) 
    {
        GSBasedINTTInnerSingle<<<num_blocks, THREADS_PER_BLOCK>>>(x_dev, q, mu, bit_length, gpu_ipowers_dev, n, n_inv, n_of_groups);
    }
  }
  HELIB_NTIMER_STOP(KernelNTT_inv);
}

void gpu_mulMod(NTL::zz_pX& x, unsigned long long x_dev[], unsigned long long gpu_powers_m_dev[], unsigned long long p, int n){
    long dx = deg(x);

  HELIB_NTIMER_START(gpu_mulMod_cuMemSet);
    cudaMemset(x_dev, 0, n*8);
  HELIB_NTIMER_STOP(gpu_mulMod_cuMemSet);

  HELIB_NTIMER_START(gpu_mulMod_cpyHD);
    CHECK(cudaMemcpy(x_dev, x.rep.data(), (dx+1)*sizeof(unsigned long long), cudaMemcpyHostToDevice));
  HELIB_NTIMER_STOP(gpu_mulMod_cpyHD);

  HELIB_NTIMER_START(gpu_mulMod_kernel);
    KernelMulMod<<<ceil(((double)dx+1)/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(x_dev, gpu_powers_m_dev, p, dx+1);
  HELIB_NTIMER_STOP(gpu_mulMod_kernel);
}

void gpu_addMod(unsigned long long x_dev[], long n, long dx, unsigned long long p){
      int n_blocks= ceil(((double)(dx+1) - n + 1) / 512);
      add_mod_kernel<<<n_blocks, 512>>>(x_dev, n, dx, p);
}

void gpu_mulMod2(NTL::zz_pX& x, unsigned long long x_dev[], unsigned long long x_pinned[], unsigned long long gpu_powers_m_dev[], unsigned long long p, int n){
	HELIB_NTIMER_START(AfterPolyMul_mulMod_kernel);
    KernelMulMod<<<ceil((double)n/THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(x_dev, gpu_powers_m_dev, p, n);
	HELIB_NTIMER_STOP(AfterPolyMul_mulMod_kernel);

	HELIB_NTIMER_START(AfterPolyMul_mulMod_cpyDH);
  CHECK(cudaMemcpy(x.rep.data(), x_dev, n*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  x.normalize();
	HELIB_NTIMER_STOP(AfterPolyMul_mulMod_cpyDH);
}

void gpu_parallel_copy(long m, unsigned long long *x_pinned, unsigned long long *x_dev, long *zMStar_gpu, NTL::vec_long& y_h, long target_dev[]){
    int numBlocks = (m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    parallel_copy<<<numBlocks, THREADS_PER_BLOCK>>>(m, x_pinned, x_dev, zMStar_gpu, target_dev);
    CHECK(cudaMemcpy(y_h.data(), x_pinned, y_h.length()*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
}
#endif //Ardhi: End of blocking function

void initializeStreams(long n_streams, std::vector<cudaStream_t> &streams){
  for(int i=0; i<n_streams; i++){
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams.push_back(stream);
  }
}


void initcuFFTBuffer(long m, cufftHandle plan, cufftDoubleComplex *buf_dev){
  CHECK(cudaMalloc(&buf_dev, m*sizeof(cufftDoubleComplex)));
}

void usecuFFT(std::vector<std::complex<double>>& buf, long m, const cufftHandle& plan, cufftDoubleComplex *buf_dev){

  int check1 = sizeof(cufftDoubleComplex);
  int check2 = sizeof(std::complex<double>);
  int check3 = buf.size();
  
  CHECK(cudaMalloc(&buf_dev, m*sizeof(cufftDoubleComplex)));

  CHECK(cudaMemcpy(buf_dev, buf.data(), m*sizeof(std::complex<double>), cudaMemcpyHostToDevice));

  CHECK_CUFFT_ERRORS(cufftExecZ2Z(plan, buf_dev, buf_dev, CUFFT_FORWARD));

  CHECK(cudaMemcpy(buf.data(), buf_dev, m*sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

}