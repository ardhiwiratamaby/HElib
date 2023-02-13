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
#ifndef HELIB_BLUESTEIN_H
#define HELIB_BLUESTEIN_H

/**
 * @file bluestein.h
 * @brief declaration of BluesteinFFT(x, n, root, powers, powers_aux, Rb):
 *
 * Compute length-n FFT of the coefficient-vector of x (in place)
 * If the degree of x is less than n then it treats the top coefficients
 * as 0, if the degree of x is more than n then the extra coefficients are
 * ignored. Similarly, if the top entries in x are zeros then x will have
 * degree smaller than n. The argument root is a 2n-th root of unity, namely
 * BluesteinFFT(...,root,...)=DFT(...,root^2,...).
 *
 * The inverse-FFT is obtained just by calling BluesteinFFT(... root^{-1}),
 * but this procedure is *NOT SCALED*, so BluesteinFFT(x,n,root,...) and
 * then BluesteinFFT(x,n,root^{-1},...) will result in x = n * x_original
 *
 * The values powers, powers_aux, and Rb must be precomputed by first
 * calling BluesteinInit(n, root, powers, powers_aux, Rb).
 *
 **/

#include <helib/NumbTh.h>
#include "cuda_runtime.h"

namespace helib {

//! @brief initialize bluestein
void BluesteinInit(long n, long k2,
                   const NTL::zz_p& root,
                   NTL::zz_pX& powers,
                   NTL::Vec<NTL::mulmod_precon_t>& powers_aux,
                   NTL::fftRep& Rb, unsigned long long RbInVec[], const NTL::zz_p& psi, NTL::zz_pX& RbInPoly, std::vector<unsigned long long>& gpu_powers, unsigned long long gpu_powers_dev[], unsigned long long gpu_powers_m_dev[], long zMStar_dev[], long zMStar_h[], long target_dev[], long target_h[]);

//! @brief apply bluestein
void BluesteinFFT(NTL::zz_pX& x,
                  long n,
                  long k2,
                  long k2_inv,
                  const NTL::zz_p& root,
                  const NTL::zz_pX& powers,
                  const NTL::Vec<NTL::mulmod_precon_t>& powers_aux,
                  const NTL::fftRep& Rb, const unsigned long long RbInVec[], unsigned long long RaInVec[], const NTL::zz_p& psi, const NTL::zz_p& inv_psi, UNUSED const NTL::zz_pX& RbInPoly, const std::vector<unsigned long long>& gpu_powers, const std::vector<unsigned long long>& gpu_ipowers, unsigned long long gpu_powers_dev[], unsigned long long gpu_ipowers_dev[], unsigned long long gpu_powers_m_dev[], unsigned long long x_dev[], unsigned long long x_pinned[], cudaStream_t stream);

} // namespace helib

#endif // ifndef HELIB_BLUESTEIN_H
