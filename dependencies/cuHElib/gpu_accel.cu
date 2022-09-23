#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include "gpu_accel.cuh"
#include <NTL/ZZVec.h>

//device variable;
long *d_A, *d_B, *d_C, *d_modulus, *d_scalar;
size_t bytes;
long d_phim, d_n_rows;

long *contiguousHostMapA, *contiguousHostMapB, *contiguousModulus, *scalarPerRow;

void InitContiguousHostMapModulus(long phim, int n_rows){
  contiguousHostMapA = (long *)malloc(phim*n_rows*sizeof(long));
  contiguousHostMapB = (long *)malloc(phim*n_rows*sizeof(long));
  contiguousModulus = (long *)malloc(n_rows*sizeof(long));
  scalarPerRow = (long *)malloc(n_rows*sizeof(long));
}

void setScalar(long index, long data){
  scalarPerRow[index] = data;
}

void setMapA(long index, long data){
  contiguousHostMapA[index] = data;
}

void setMapB(long index, long data){
  contiguousHostMapB[index] = data;
}

long getMapA(long index){
  return contiguousHostMapA[index];
}

long getMapB(long index){
  return contiguousHostMapB[index];
}

void setModulus(long index, long data){
  contiguousModulus[index] = data;
}

void InitGPUBuffer(long phim, int n_rows){
  d_phim = phim;
  d_n_rows = n_rows;

  bytes = phim*n_rows*sizeof(long);
  // Allocate memory for arrays d_A, d_B, and d_C on device
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);
  cudaMalloc(&d_modulus, d_n_rows*sizeof(long));
  cudaMalloc(&d_scalar, d_n_rows*sizeof(long));
}

void DestroyGPUBuffer(){
    // Free GPU memory
    cudaFree(d_C);
    cudaFree(d_A);
    cudaFree(d_B);
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
void CudaEltwiseAddMod(){
    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, contiguousHostMapA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, contiguousHostMapB, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_modulus, contiguousModulus, bytes, cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(d_phim*d_n_rows) / thr_per_blk );

    // Launch kernel
    kernel_addMod<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, d_phim*d_n_rows, d_modulus, d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    cudaMemcpy(contiguousHostMapA, d_C, bytes, cudaMemcpyDeviceToHost);


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

void CudaEltwiseAddMod(long scalar){
    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, contiguousHostMapA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scalar, scalarPerRow, d_n_rows*sizeof(long), cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(d_phim*d_n_rows) / thr_per_blk );

    // Launch kernel
    kernel_addModScalar<<< blk_in_grid, thr_per_blk >>>(d_A, d_scalar, d_C, d_phim*d_n_rows, d_modulus, d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    cudaMemcpy(contiguousHostMapA, d_C, bytes, cudaMemcpyDeviceToHost);

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

void CudaEltwiseSubMod(){
    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, contiguousHostMapA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, contiguousHostMapB, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_modulus, contiguousModulus, bytes, cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(d_phim*d_n_rows) / thr_per_blk );

    // Launch kernel
    kernel_subMod<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, d_phim*d_n_rows, d_modulus, d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    cudaMemcpy(contiguousHostMapA, d_C, bytes, cudaMemcpyDeviceToHost);

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

void CudaEltwiseSubMod(long scalar){
    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, contiguousHostMapA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scalar, scalarPerRow, d_n_rows*sizeof(long), cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(d_phim*d_n_rows) / thr_per_blk );

    // Launch kernel
    kernel_subModScalar<<< blk_in_grid, thr_per_blk >>>(d_A, d_scalar, d_C, d_phim*d_n_rows, d_modulus, d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    cudaMemcpy(contiguousHostMapA, d_C, bytes, cudaMemcpyDeviceToHost);

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
    __int128 temp_res=0;
    __int128 temp_a=0;
    __int128 temp_b=0;

    temp_a = a[tid];
    temp_b = b[tid];

    temp_res = temp_a * temp_b;
    // temp_res = temp_res%modulus;
    temp_res = myMod2(temp_res, d_modulus[tid/phim]);

    // d_result[tid] = temp_res;
    // result[tid] %= modulus;
    result[tid]=temp_res;

    // result[tid]=mul_mod(a[tid], b[tid], modulus);
  }
}

void CudaEltwiseMultMod(){
    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, contiguousHostMapA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, contiguousHostMapB, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_modulus, contiguousModulus, bytes, cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(d_phim*d_n_rows) / thr_per_blk );

    // Launch kernel
    kernel_mulMod<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, d_phim*d_n_rows, d_modulus, d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    cudaMemcpy(contiguousHostMapA, d_C, bytes, cudaMemcpyDeviceToHost);

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

void CudaEltwiseMultMod(long scalar){
   // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, contiguousHostMapA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scalar, scalarPerRow, d_n_rows*sizeof(long), cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(d_phim*d_n_rows) / thr_per_blk );

    // Launch kernel
    kernel_mulModScalar<<< blk_in_grid, thr_per_blk >>>(d_A, d_scalar, d_C, d_phim*d_n_rows, d_modulus, d_phim);
    // cudaDeviceSynchronize();

    // Copy data from device array d_C to host array result
    cudaMemcpy(contiguousHostMapA, d_C, bytes, cudaMemcpyDeviceToHost);

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