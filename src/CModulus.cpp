/* Copyright (C) 2012-2020 IBM Corp.
 * This program is Licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. See accompanying LICENSE file.
 */

/* Intel HEXL integration.
 * Copyright (C) 2021 Intel Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *  http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* CModulus.cpp - supports forward and backward length-m FFT transformations
 *
 * This is a wrapper around the bluesteinFFT routines, for one modulus q.
 *
 * On initialization, it initializes NTL's zz_pContext for this q
 * and computes a 2m-th root of unity r mod q and also r^{-1} mod q.
 * Thereafter this class provides FFT and iFFT routines that converts between
 * time & frequency domains. Some tables are computed the first time that
 * each directions is called, which are then used in subsequent computations.
 *
 * The "time domain" polynomials are represented as ZZX, which are reduced
 * modulo Phi_m(X). The "frequency domain" are just vectors of integers
 * (vec_long), that store only the evaluation in primitive m-th
 * roots of unity.
 */
#include <helib/CModulus.h>
#include <helib/timing.h>

#ifdef USE_INTEL_HEXL
#include "intelExt.h"
#endif

namespace helib {

// It is assumed that m,q,context, and root are already set. If root is set
// to zero, it will be computed by the compRoots() method. Then rInv is
// computed as the inverse of root.

NTL::zz_pContext BuildContext(long p, long maxroot)
{
  if (maxroot <= NTL::CalcMaxRoot(p))
    return NTL::zz_pContext(NTL::INIT_USER_FFT, p);
  else
    return NTL::zz_pContext(p, maxroot);
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


// Constructor: it is assumed that zms is already set with m>1
// If q == 0, then the current context is used
Cmodulus::Cmodulus(const PAlgebra& zms, long qq, long rt) :
    q(qq), zMStar(&zms), root(rt)
{
  assertTrue<InvalidArgument>(zms.getM() > 1,
                              "Bad Z_m^* modulus m (must be greater than 1)");
  bool explicitModulus = true;

  // Correction for q == 0
  if (this->q == 0) {
    this->q = NTL::zz_p::modulus();
    explicitModulus = false;
  }

  qinv = NTL::PrepMulMod(q);

  long mm = zms.getM();
  m_inv = NTL::InvMod(mm, q);

  NTL::zz_pBak bak;

  if (zms.getPow2()) {
    // special case when m is a power of 2

    assertTrue(explicitModulus,
               "bad non explicit q value (cannot be non explicit when m "
               "is a power of 2)");

    bak.save();

    RandomState state;
    SetSeed(NTL::conv<NTL::ZZ>("84547180875373941534287406458029"));
    // DIRT: this ensures the roots are deterministically generated
    //    inside the zz_pContext constructor
    context = NTL::zz_pContext(NTL::INIT_USER_FFT, q);
    state.restore();

    context.restore();

    powers.reset(new NTL::zz_pX);
    ipowers.reset(new NTL::zz_pX);

    long k = zms.getPow2();
    long phim = 1L << (k - 1);

    assertTrue(k <= NTL::zz_pInfo->MaxRoot,
               "Roots count exceeds maximum rootTables size (m = 2^k && k > "
               "zz_pInfo->Maxroot && rootTables are 0..zz_pInfo->Maxroot)");
    // rootTables get initialized 0..zz_pInfo->Maxroot

#ifdef HELIB_OPENCL
    altFFTInfo = MakeSmart<AltFFTPrimeInfo>();
    InitAltFFTPrimeInfo(*altFFTInfo, *zz_pInfo->p_info, k - 1);
#endif

    long w0 = NTL::zz_pInfo->p_info->RootTable[0][k];
    long w1 = NTL::zz_pInfo->p_info->RootTable[1][k];

    powers->rep.SetLength(phim);
    powers_aux.SetLength(phim);
    for (long i = 0, w = 1; i < phim; i++) {
      powers->rep[i] = w;
      powers_aux[i] = NTL::PrepMulModPrecon(w, q);
      w = NTL::MulMod(w, w0, q);
    }

    ipowers->rep.SetLength(phim);
    ipowers_aux.SetLength(phim);
    for (long i = 0, w = 1; i < phim; i++) {
      ipowers->rep[i] = w;
      ipowers_aux[i] = NTL::PrepMulModPrecon(w, q);
      w = NTL::MulMod(w, w1, q);
    }

    return;
  }

  if (explicitModulus) {
    bak.save(); // backup the current modulus
    context = BuildContext(q, zms.fftSizeNeeded());
    // fftSizeNeeded() returns next power of 2 after 2m
    context.restore(); // set NTL's current modulus to q
  } else
    context.save();

  if (root == 0) { // Find a 2m-th root of unity modulo q, if not given
    NTL::zz_p rtp;
    long m = zms.getM();
    long e;
    if (m % 2 == 0)
      e = 2 * m;
    else
      e = m;
    // for odd m, we can use a primitive m'th roots of unity,
    // which will facilitate the use of a truncated FFT
    // in the Bluestein FFT implementation

    FindPrimitiveRoot(rtp, e); // NTL routine, relative to current modulus
    if (rtp == 0)              // sanity check
      throw RuntimeError("Cmod::compRoots(): no 2m'th roots of unity mod q");
    root = NTL::rep(rtp);
  }
  rInv = NTL::InvMod(root, q); // set rInv = root^{-1} mod q

  // Allocate memory (relative to current modulus that was defined above).
  // These objects will be initialized when anyone calls FFT/iFFT.

  NTL::zz_pX phimx_poly;
  conv(phimx_poly, zms.getPhimX());

  powers.reset(new NTL::zz_pX);
  Rb.reset(new NTL::fftRep);
  ipowers.reset(new NTL::zz_pX);
  iRb.reset(new NTL::fftRep);
  phimx.reset(new zz_pXModulus1(zms.getM(), phimx_poly));

  // RbInVec.reset(new NTL::vec_zz_p);

  // iRbInVec.reset(new NTL::vec_zz_p);
  RbInPoly.reset(new NTL::zz_pX);
  iRbInPoly.reset(new NTL::zz_pX);
  gpu_powers.reset(new std::vector<unsigned long long>);
  gpu_ipowers.reset(new std::vector<unsigned long long>);

  // myPsi.reset(new NTL::vec_zz_p);

  // psi = 0;
  // psi.reset(new NTL::zz_p);
  // psi_inv.reset(new NTL::zz_p);

  //Ardhi: get nth-root of unity for n=2^k and current modulus
  long k = NTL::NextPowerOfTwo(2 * mm - 1);
  k2 = 1L << k; // k2 = 2^k
  NTL::zz_p rtp;
  FindPrimitiveRoot(rtp, k2); // NTL routine, relative to current modulus
  if (rtp == 0)              // sanity check
    throw RuntimeError("Cmod::compRoots(): no 2^k'th roots of unity mod q");
  long k2_root = NTL::rep(rtp);

  //Ardhi: get psi, psi=root^(1/2)
  // long psi;
  long p = NTL::zz_p::modulus();
  NTL::ZZ temp = NTL::SqrRootMod(NTL::conv<NTL::ZZ>(k2_root), NTL::conv<NTL::ZZ>(p));
  NTL::conv(psi, temp);
  // myPsi[0] = psi;
  //Ardhi: get inverse psi
  long inv_psi = NTL::InvMod(rep(psi), p);
  psi_inv = inv_psi;
  k2_inv = NTL::InvMod(k2, p);

  gpu_powers->reserve(k2);
  gpu_ipowers->reserve(k2);

  cudaMalloc(&RbInVec, k2 * sizeof(unsigned long long));
  cudaMalloc(&iRbInVec, k2 * sizeof(unsigned long long));  
  cudaMalloc(&RaInVec, k2 * sizeof(unsigned long long));
  cudaMalloc(&iRaInVec, k2 * sizeof(unsigned long long));
  cudaMalloc(&gpu_powers_dev, k2 * sizeof(unsigned long long));
  cudaMalloc(&gpu_ipowers_dev, k2 * sizeof(unsigned long long));
  cudaMalloc(&zMStar_dev, mm * sizeof(long));
  cudaMalloc(&target_dev, mm * sizeof(long));

  CHECK(cudaMalloc(&gpu_powers_m_dev, mm * sizeof(unsigned long long)));
  CHECK(cudaMalloc(&gpu_ipowers_m_dev, mm * sizeof(unsigned long long)));
  CHECK(cudaMalloc(&x_dev, k2 * sizeof(unsigned long long)));
  // CHECK(cudaMallocHost(&x_pinned, mm * sizeof(unsigned long long)));
  CHECK(cudaMalloc(&x_pinned, mm * sizeof(unsigned long long)));

  //Ardhi: generate twiddle factors for gpu ntt with n=2^k
  for (unsigned int i = 0; i < k2; i++)
  {
      gpu_powers->push_back(NTL::PowerMod(rep(psi), bitReverse(i, std::log2(k2)), p));
      gpu_ipowers->push_back(NTL::PowerMod(inv_psi, bitReverse(i, std::log2(k2)), p));

      // gpu_powers[i] = modpow64(rep(psi), bitReverse(i, std::log2(k2)), p);
      // gpu_ipowers[i] = NTL::PowerMod(inv_psi, bitReverse(i, std::log2(k2)), p);

  }

  zMStar_h = (long *)malloc(mm*sizeof(long));
  target_h = (long *)malloc(mm*sizeof(long));

  int sum=0;
  for(int i=0; i<mm; i++){
    zMStar_h[i] = zMStar->inZmStar(i);
    target_h[i] = sum;
    sum += zMStar_h[i];
  }

  // cudaMemcpy(gpu_powers_dev, gpu_powers->data(), k2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
  // cudaMemcpy(gpu_ipowers_dev, gpu_ipowers->data(), k2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);

  BluesteinInit(mm, k2, NTL::conv<NTL::zz_p>(root), *powers, powers_aux, *Rb, RbInVec, psi, *RbInPoly, *gpu_powers, gpu_powers_dev, gpu_powers_m_dev, zMStar_dev, zMStar_h, target_dev, target_h);
  BluesteinInit(mm, k2, NTL::conv<NTL::zz_p>(rInv), *ipowers, ipowers_aux, *iRb, iRbInVec, psi_inv, *iRbInPoly, *gpu_ipowers, gpu_ipowers_dev, gpu_ipowers_m_dev, zMStar_dev, zMStar_h, target_dev, target_h);
}


Cmodulus& Cmodulus::operator=(const Cmodulus& other)
{
  if (this == &other)
    return *this;

  zMStar = other.zMStar; // Yes, really copy this pointer
  q = other.q;
  qinv = other.qinv;
  m_inv = other.m_inv;

  context = other.context;
  NTL::zz_pBak bak;
  bak.save();        // backup the current modulus
  context.restore(); // Set NTL's current modulus to q

  // NOTE: newer versions of NTL allow fftRep's and zz_pXModulus's to be copied
  // "out of context" (versions after 7.0.*). However, those copies
  // are not intended to allow copies out of one context into another,
  // so we still need to use copied_ptr's (but not context restoration).
  // All of this is fairly academic, as I don't think we really
  // copy Contexts around anywhere. Also, it would be cleaner
  // to make the vector in Context be a vector of copied_ptr<Cmodulus>

  root = other.root;
  rInv = other.rInv;

  powers_aux = other.powers_aux;
  ipowers_aux = other.ipowers_aux;

  // copy data, not pointers in these fields
  powers = other.powers;
  Rb = other.Rb;
  RbInVec = other.RbInVec;
  ipowers = other.ipowers;
  iRb = other.iRb;
  iRbInVec = other.iRbInVec;
  phimx = other.phimx;
  psi = other.psi;
  psi_inv = other.psi_inv;
  RbInPoly = other.RbInPoly;
  iRbInPoly = other.iRbInPoly;
  gpu_powers = other.gpu_powers;
  gpu_ipowers = other.gpu_ipowers;

  gpu_powers_dev = other.gpu_powers_dev;
  gpu_ipowers_dev = other.gpu_ipowers_dev;
  gpu_powers_m_dev = other.gpu_powers_m_dev;
  gpu_ipowers_m_dev = other.gpu_ipowers_m_dev;
  x_dev = other.x_dev;
  x_pinned = other.x_pinned;
  k2 = other.k2;
  k2_inv = other.k2_inv;
  zMStar_dev = other.zMStar_dev;
  zMStar_h = other.zMStar_h;
  target_dev = other.target_dev;
  target_h = other.target_h;
  // myPsi = other.myPsi;

#ifdef HELIB_OPENCL
  altFFTInfo = other.altFFTInfo;
#endif

  return *this;
}

//==================================================================
// Starting with NTL 11.1.0, the NTL FFT routines do not do any bit reversal.
// Specifically, FFTFwd leaves its outputs bit reversed, and FFTInv1
// requires that its inputs are already bit reversed.
// These low-level routines are only used when m is a power of two.
// As a work-around, we borrow NTL's old bit reversal code,
// but only when NTL_PROVIDES_TRUNC_FFT is defined (this macro is defined
// in the newer versions of NTL that do not do the bit reversal).
// So this HElib code should be compatible with NTL versions both before
// and after version 11.1.0.

#ifdef NTL_PROVIDES_TRUNC_FFT

static long RevInc(long a, long k)
{
  long j = k;
  long m = 1L << (k - 1);

  while (j && (m & a)) {
    a ^= m;
    m >>= 1;
    j--;
  }
  if (j)
    a ^= m;
  return a;
}

// FIXME: This could potentially be shared across threads, using
// a "lazy table".
static inline NTL::Vec<long>* get_brc_mem()
{
  using namespace NTL;
  NTL_TLS_LOCAL_INIT(NTL::Vec<NTL::Vec<long>>,
                     brc_mem_vec,
                     (NTL::INIT_SIZE, NTL_FFTMaxRoot + 1));
  return brc_mem_vec.elts();
}

#define NTL_BRC_THRESH (11)
#define NTL_BRC_Q (5)

// Must have NTL_BRC_THRESH >= 2*NTL_BRC_Q
// Should also have (1L << (2*NTL_BRC_Q)) small enough
// so that we can fit that many longs into the cache

static long* BRC_init(long k)
{
  NTL::Vec<long>* brc_mem = get_brc_mem();

  long n = (1L << k);
  brc_mem[k].SetLength(n);
  long* rev = brc_mem[k].elts();
  for (long i = 0, j = 0; i < n; i++, j = RevInc(j, k))
    rev[i] = j;
  return rev;
}

static void BasicBitReverseCopy(long* NTL_RESTRICT B,
                                const long* NTL_RESTRICT A,
                                long k)
{
  NTL::Vec<long>* brc_mem = get_brc_mem();

  long n = 1L << k;
  long* NTL_RESTRICT rev;

  rev = brc_mem[k].elts();
  if (!rev)
    rev = BRC_init(k);

  for (long i = 0; i < n; i++)
    B[rev[i]] = A[i];
}

static void COBRA(long* NTL_RESTRICT B, const long* NTL_RESTRICT A, long k)
{
  NTL::Vec<long>* brc_mem = get_brc_mem();

  using namespace NTL;
  NTL_TLS_LOCAL(NTL::Vec<long>, BRC_temp);

  long q = NTL_BRC_Q;
  long k1 = k - 2 * q;
  long* NTL_RESTRICT rev_k1;
  long* NTL_RESTRICT rev_q;
  long* NTL_RESTRICT T;
  long a1, b1, c1;

  rev_k1 = brc_mem[k1].elts();
  if (!rev_k1)
    rev_k1 = BRC_init(k1);

  rev_q = brc_mem[q].elts();
  if (!rev_q)
    rev_q = BRC_init(q);

  T = BRC_temp.elts();
  if (!T) {
    BRC_temp.SetLength(1L << (2 * q));
    T = BRC_temp.elts();
  }

  for (long b = 0; b < (1L << k1); b++) {
    b1 = rev_k1[b];
    for (long a = 0; a < (1L << q); a++) {
      a1 = rev_q[a];
      for (long c = 0; c < (1L << q); c++)
        T[(a1 << q) + c] = A[(a << (k1 + q)) + (b << q) + c];
    }

    for (long c = 0; c < (1L << q); c++) {
      c1 = rev_q[c];
      for (long a1 = 0; a1 < (1L << q); a1++)
        B[(c1 << (k1 + q)) + (b1 << q) + a1] = T[(a1 << q) + c];
    }
  }
}

static void BitReverseCopy(long* NTL_RESTRICT B,
                           const long* NTL_RESTRICT A,
                           long k)
{
  if (k <= NTL_BRC_THRESH)
    BasicBitReverseCopy(B, A, k);
  else
    COBRA(B, A, k);
}
#endif

//================================================

void Cmodulus::FFT_aux(NTL::vec_long& y, NTL::zz_pX& tmp, cudaStream_t stream) const
{
  HELIB_TIMER_START;

  if (zMStar->getPow2()) {
    // Special case: m is a power of 2

    long k = zMStar->getPow2();
    long phim = (1L << (k - 1));
    long dx = deg(tmp);
    long p = NTL::zz_p::modulus();

    y.SetLength(phim);
    long* yp = y.elts();

    const NTL::zz_p* tmp_p = tmp.rep.elts();

#ifdef USE_INTEL_HEXL

    for (long i = 0; i <= dx; ++i) {
      yp[i] = rep(tmp_p[i]);
    }

    for (long i = dx + 1; i < phim; ++i) {
      yp[i] = 0;
    }

    intel::FFTFwd(yp, yp, phim, p);

#else

    const NTL::zz_p* powers_p = (*powers).rep.elts();
    const NTL::mulmod_precon_t* powers_aux_p = powers_aux.elts();

    for (long i = 0; i <= dx; i++) {
      yp[i] = NTL::MulModPrecon(rep(tmp_p[i]),
                                rep(powers_p[i]),
                                p,
                                powers_aux_p[i]);
    }

    for (long i = dx + 1; i < phim; i++) {
      yp[i] = 0;
    }

#ifdef HELIB_OPENCL
    AltFFTFwd(yp, yp, k - 1, *altFFTInfo);
#else

#ifndef NTL_PROVIDES_TRUNC_FFT
    NTL::FFTFwd(yp, yp, k - 1, *NTL::zz_pInfo->p_info);
#else
    NTL::FFTFwd(yp, yp, k - 1, *NTL::zz_pInfo->p_info);
#endif

#endif // HELIB_OPENCL

#endif // USE_INTEL_HEXL

    // Now we have to bit reverse the result
    // The BitReverseCopy routine does not allow aliasing, so
    // we have to do an extra copy here.
    // We use the fact tmp1 and y do not alias.
    NTL::vec_long& bit_reversed = Cmodulus::getScratch_vec_long();
    bit_reversed.SetLength(phim);
    long* bit_reversed_p = bit_reversed.elts();

    BitReverseCopy(bit_reversed_p, yp, k - 1);
    std::copy_n(bit_reversed_p, phim, yp);

    return;
  }

  NTL::zz_p rt;
  conv(rt, root); // convert root to zp format

  //Ardhi: I want to replace the BluesteinFFT with BluesteinGPU

  // std::cout<<"Tmp degree: "<<deg(tmp)<<"\nBefore BluesteinFFT\n";
  // for (int i = 0; i <= deg(tmp); ++i) std::cout<<tmp[i]<<", ";
  // std::cout<<std::endl;


  // call the FFT routine
  BluesteinFFT(tmp, getM(), k2, k2_inv, rt, *powers, powers_aux, *Rb, RbInVec, RaInVec, psi, psi_inv, *RbInPoly, *gpu_powers, *gpu_ipowers, gpu_powers_dev, gpu_ipowers_dev, gpu_powers_m_dev, x_dev, x_pinned, stream);

  // copy the result to the output vector y, keeping only the
  // entries corresponding to primitive roots of unity
  // y.SetLength(zMStar->getPhiM());

  gpu_parallel_copy(long(this->getM()), x_pinned, x_dev, zMStar_dev, y, target_dev, stream);

#if 0
	HELIB_NTIMER_START(FFT_aux_copyresult);
  for (long i = 0, j = 0; i < long(this->getM()); i++){
#if 0
    // printf("cpu zMStar %ld\n", zMStar_h[i]);
    if(zMStar->inZmStar(i) != zMStar_h[i])
      throw std::runtime_error("zMStar_h missmatch with the default");
#endif
    if (zMStar->inZmStar(i)){
      y[j++] = rep(coeff(tmp, i));
      if(tmp.rep[j-1] != y[j-1])
        std::cout<<"missmatch with CPU.\ny["<<j-1<<"]: "<<tmp.rep[j-1]<<" y_dev["<<j-1<<"]: "<<y[j-1]<<"zMStar["<<i<<"]: "<<zMStar->inZmStar(i)<<std::endl;
    }
  }
	HELIB_NTIMER_STOP(FFT_aux_copyresult);
#endif

}

void Cmodulus::FFT(NTL::vec_long& y, const NTL::ZZX& x, cudaStream_t stream) const
{
  HELIB_TIMER_START;
  NTL::zz_pBak bak;
  bak.save();
  context.restore();

  // NTL::zz_pX& tmp = Cmodulus::getScratch_zz_pX();
  NTL::zz_pX tmp;
  {
    HELIB_NTIMER_START(FFT_remainder);
    convert(tmp, x); // convert input to zpx format
  }
  
  FFT(y, tmp, stream);
}

void Cmodulus::FFT(NTL::vec_long& y, const zzX& x, cudaStream_t stream) const
{
  HELIB_TIMER_START;
  NTL::zz_pBak bak;
  bak.save();
  context.restore();

  // NTL::zz_pX& tmp = Cmodulus::getScratch_zz_pX();
  NTL::zz_pX tmp;
  {
    HELIB_NTIMER_START(FFT_remainder);
    convert(tmp, x); // convert input to zpx format
  }

  FFT(y, tmp, stream);
}

inline void checkCudaError(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        printf("%s\n", msg);
        printf("Error: %s\n", cudaGetErrorString(status));
        // exit(EXIT_FAILURE);
    }
}

void Cmodulus::FFT(NTL::vec_long& y, NTL::zz_pX& x, cudaStream_t stream) const
{
  HELIB_TIMER_START;
  NTL::zz_pBak bak;
  bak.save();
  context.restore();
  
  
  x.SetLength(getM());
  
  FFT_aux(y, x, stream);
}

void Cmodulus::iFFT(NTL::zz_pX& x, const NTL::vec_long& y) const
{
  HELIB_TIMER_START;
  NTL::zz_pBak bak;
  bak.save();
  context.restore();

  if (zMStar->getPow2()) {
    // special case when m is a power of 2

    long k = zMStar->getPow2();
    long phim = (1L << (k - 1));
    long p = NTL::zz_p::modulus();

    const long* yp = y.elts();

    NTL::vec_long& tmp = Cmodulus::getScratch_vec_long();
    tmp.SetLength(phim);
    long* tmp_p = tmp.elts();

    // We have to bit reverse the inputs to FFTRev1
    // The BitReverseCopy routine does not allow aliasing.
    // We use the fact that y and tmp do not alias

    BitReverseCopy(tmp_p, yp, k - 1);

#ifdef USE_INTEL_HEXL

    intel::FFTRev1(tmp_p, tmp_p, phim, p);

    x.rep.SetLength(phim);
    NTL::zz_p* xp = x.rep.elts();

    for (long i = 0; i < phim; ++i) {
      xp[i].LoopHole() = tmp_p[i];
    }

#else

    const NTL::zz_p* ipowers_p = (*ipowers).rep.elts();
    const NTL::mulmod_precon_t* ipowers_aux_p = ipowers_aux.elts();

#ifdef HELIB_OPENCL
    AltFFTRev1(tmp_p, yp, k - 1, *altFFTInfo);
#else

#ifndef NTL_PROVIDES_TRUNC_FFT
    NTL::FFTRev1(tmp_p, yp, k - 1, *NTL::zz_pInfo->p_info);
#else
    NTL::FFTRev1(tmp_p, tmp_p, k - 1, *NTL::zz_pInfo->p_info);
#endif

#endif // HELIB_OPENCL

    x.rep.SetLength(phim);
    NTL::zz_p* xp = x.rep.elts();

    for (long i = 0; i < phim; ++i) {
      xp[i].LoopHole() =
          NTL::MulModPrecon(tmp_p[i], rep(ipowers_p[i]), p, ipowers_aux_p[i]);
    }

#endif // USE_INTEL_HEXL

    x.normalize();

    return;
  } // End of special case if m is power of 2

  NTL::zz_p rt;
  long m = getM();

  // convert input to zpx format, initializing only the coeffs i s.t. (i,m)=1
  x.rep.SetLength(m);
  for (long i = 0, j = 0; i < m; i++)
    if (zMStar->inZmStar(i))
      x.rep[i].LoopHole() = y[j++]; // DIRT: y[j] already reduced
  x.normalize();
  conv(rt, rInv); // convert rInv to zp format

  BluesteinFFT(x, m, k2, k2_inv, rt, *ipowers, ipowers_aux, *iRb, iRbInVec, iRaInVec, psi_inv, psi, *iRbInPoly, *gpu_ipowers, *gpu_powers, gpu_ipowers_dev, gpu_powers_dev, gpu_ipowers_m_dev, x_dev, x_pinned); // call the FFT routine
}

NTL::zz_pX& Cmodulus::getScratch_zz_pX()
{
  NTL_THREAD_LOCAL static NTL::zz_pX scratch;
  return scratch;
}

NTL::Vec<long>& Cmodulus::getScratch_vec_long()
{
  NTL_THREAD_LOCAL static NTL::Vec<long> scratch;
  return scratch;
}

NTL::fftRep& Cmodulus::getScratch_fftRep(long k)
{
  NTL_THREAD_LOCAL static NTL::fftRep rep;
  NTL_THREAD_LOCAL static long MaxK[4] = {-1, -1, -1, -1};

  long NumPrimes = NTL::zz_pInfo->NumPrimes;

  for (long i = 0; i < NumPrimes; i++) {
    if (k > MaxK[i]) {
      rep.tbl[i].SetLength(1L << k);
      MaxK[i] = k;
    }
  }

  rep.NumPrimes = NumPrimes;
  rep.k = rep.MaxK = k;

  return rep;
}

} // namespace helib
