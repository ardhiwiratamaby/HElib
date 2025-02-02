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

#ifndef HELIB_CMODULUS_H
#define HELIB_CMODULUS_H
/**
 * @file CModulus.h
 * @brief Supports forward and backward length-m FFT transformations
 *
 * This is a wrapper around the bluesteinFFT routines, for one modulus q.
 **/
#include <helib/NumbTh.h>
#include <helib/PAlgebra.h>
#include <helib/bluestein.h>
#include <helib/ClonedPtr.h>
#include "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/dependencies/cuHElib/gpu_accel.cuh"
#include "cuda_runtime.h"

namespace helib {

/**
 * @class Cmodulus
 * @brief Provides FFT and iFFT routines modulo a single-precision prime
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
 **/
class Cmodulus
{
private:
  //! The modulus
  long q;
  //! PrepMulMod(q);
  NTL::mulmod_t qinv;

  //! NTL's tables for this modulus
  NTL::zz_pContext context;

  //! Points to the Zm* structure, m is FFT size
  const PAlgebra* zMStar;

  //! m^{-1} mod q
  long m_inv;

  //! 2m-th root of unity modulo q
  long root;
  //! root^{-1} mod q
  long rInv;

  //2k-root^1/2 for GPU NTT// Need to store psi since we may have different psi for the same configuration //store it in zz_p to support object functionality
  // CopiedPtr<NTL::zz_p> psi;
  // CopiedPtr<NTL::zz_p> psi_inv;  
  NTL::zz_p psi;
  NTL::zz_p psi_inv;
  long k2_inv;
  long k2;
  //Ardhi: tables for forward and backward GPU NTT
  CopiedPtr<std::vector<unsigned long long>> gpu_powers;
  CopiedPtr<std::vector<unsigned long long>> gpu_ipowers;

  // tables for forward FFT
  CopiedPtr<NTL::zz_pX> powers;
  CopiedPtr<NTL::zz_pX> RbInPoly;
  NTL::Vec<NTL::mulmod_precon_t> powers_aux;
  CopiedPtr<NTL::fftRep> Rb;
  // CopiedPtr<NTL::vec_zz_p> RbInVec;
  unsigned long long *RbInVec;
  unsigned long long *RaInVec;

  unsigned long long *x_dev;
  unsigned long long *x_pinned;

  unsigned long long *gpu_powers_dev;
  unsigned long long *gpu_ipowers_dev;

  unsigned long long *gpu_powers_m_dev;
  unsigned long long *gpu_ipowers_m_dev;
  long *zMStar_dev;
  long *zMStar_h;
  long *target_dev;
  long *target_h;
  // CopiedPtr<NTL::vec_zz_p> myPsi;

  // tables for backward FFT
  CopiedPtr<NTL::zz_pX> ipowers;
  CopiedPtr<NTL::zz_pX> iRbInPoly;
  NTL::Vec<NTL::mulmod_precon_t> ipowers_aux;
  CopiedPtr<NTL::fftRep> iRb;
  // CopiedPtr<NTL::vec_zz_p> iRbInVec;
  unsigned long long *iRbInVec;
  unsigned long long *iRaInVec;

  // PhimX modulo q, for faster division w/ remainder
  CopiedPtr<zz_pXModulus1> phimx;

  // Allocate memory and compute roots
  void privateInit(const PAlgebra&, long rt);

  // auxiliary routine used by the two FFT routines
  void FFT_aux(NTL::vec_long& y, NTL::zz_pX& tmp, cudaStream_t stream) const;

public:
#ifdef HELIB_OPENCL
  SmartPtr<AltFFTPrimeInfo> altFFTInfo;
  // We need to allow copying...the underlying object
  // is immutable
#endif

  //! Default constructor
  Cmodulus() = default;

  /**
   * @brief Constructor
   * @note Specify m and q, and optionally also the root if q == 0, then the
   * current context is used
   */
  Cmodulus(const PAlgebra& zms, long qq, long rt);

  //! Copy constructor
  Cmodulus(const Cmodulus& other) { *this = other; };

  //! Copy assignment operator
  Cmodulus& operator=(const Cmodulus& other);

  // utility methods

  const PAlgebra& getZMStar() const { return *zMStar; }
  unsigned long getM() const { return zMStar->getM(); }
  unsigned long getPhiM() const { return zMStar->getPhiM(); }
  long getQ() const { return q; }
  long getMInv() const {return m_inv;}
  NTL::mulmod_t getQInv() const { return qinv; }
  long getRoot() const { return root; }
  const zz_pXModulus1& getPhimX() const { return *phimx; }
  const NTL::zz_pContext getCmodulusContext() const {return context;}
  //! @brief Restore NTL's current modulus
  void restoreModulus() const { context.restore(); }

  // FFT routines

  // sets zp context internally
  // y = FFT(x)
  void FFT(NTL::vec_long& y, const NTL::ZZX& x, cudaStream_t stream) const;
  // y = FFT(x)
  void FFT(NTL::vec_long& y, const zzX& x, cudaStream_t stream) const;
  // y = FFT(x)
  void FFT(NTL::vec_long& y, NTL::zz_pX& x, cudaStream_t stream) const;


  // expects zp context to be set externally
  // x = FFT^{-1}(y)
  void iFFT(NTL::zz_pX& x, const NTL::vec_long& y) const;

  // returns thread-local scratch space
  // DIRT: this zz_pX is used for several zz_p moduli,
  // which is not officially sanctioned by NTL, but should be OK.
  static NTL::zz_pX& getScratch_zz_pX();

  static NTL::Vec<long>& getScratch_vec_long();

  // returns thread-local scratch space
  // DIRT: this use a couple of internal, undocumented
  // NTL interfaces
  static NTL::fftRep& getScratch_fftRep(long k);
};

} // namespace helib

#endif // ifndef HELIB_CMODULUS_H
