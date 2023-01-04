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
/* bluestein.cpp -
 * An implementation of non-power-of-two FFT using Bluestein's trick
 *
 */

#include <helib/bluestein.h>
#include <helib/timing.h>
#include <helib/CModulus.h>
#include <helib/apiAttributes.h>
#include "/home/ardhy/Documents/research/new_project/bgv-comparison/HElib/dependencies/cuHElib/gpu_accel.cuh"

#define NEW_BLUE (1)

namespace helib {

/************************************************************

Victor says: I really need to document the optimizations I made
to the Bluestein logic in relation to the truncated FFT.

In the mean time, here are two emails I wrote that explain the
situation (from June 7, 2018):


What we are really computing is f*g mod x^m-1, where f and g have degree less
than m.

The way it's done now is that we replace f by a larger polynomial F, and then
compute F*g mod x^N-1, where N is the next power of 2 after 2*m-1. This is done
implicitly in NTL's FFT routine.

With the truncated FFT, a better way is as follows. Just compute the polynomial
h = f*g, which we can do with the same size FFT, but truncated to 2*m-1 terms.
Then compute h mod x^m-1 separately, which takes negligible time.


..........................................

There are some complications with the idea that I had, because of the way we
currently implement Bluestein.  For my idea to work, I need convolutions modulo
m, but that is not what we currently have.

It relates to the fact that we are working with roots of order 2*m, rather than
m.  In fact, if m is odd (which it almost always is, unless it's a power of 2),
there is no need to do this.

First, you can look here to refresh your memory on Bluestein:

https://www.dsprelated.com/freebooks/mdft/Bluestein_s_FFT_Algorithm.html

Now, the problem is that in these formulas, we work with W^{1/2}, which is the
root of order 2*m.  But if W has order m and m is itself odd, then 2 has an
inverse mod m, and so we can just work with W.  In all of the computations, we
are just computing W^{(1/2)*i} for various values of i, and so in the exponent
we're just doing arithmetic mod m.

To get the speedup using the truncated FFT, I really need these two be circular
convolutions modulo m, not modulo 2*m.

I never really understood why we needed to work with a root of order 2*m, and
now I see that we don't, at least when m is odd.


*************************************************************/

void BluesteinInit(long n,
                   const NTL::zz_p& root,
                   NTL::zz_pX& powers,
                   NTL::Vec<NTL::mulmod_precon_t>& powers_aux,
                   NTL::fftRep& Rb, NTL::vec_zz_p& RbInVec)
{
  long p = NTL::zz_p::modulus();

  NTL::zz_p one;
  one = 1;
  powers.SetMaxLength(n);

  long e;
  if (n % 2 == 0)
    e = 2 * n;
  else
    e = n;

  SetCoeff(powers, 0, one);
  for (long i = 1; i < n; i++) {
    long iSqr = NTL::MulMod(i, i, e);       // i^2 mod 2n
    SetCoeff(powers, i, power(root, iSqr)); // powers[i] = root^{i^2}
  }

  // powers_aux tracks powers
  powers_aux.SetLength(n);
  for (long i = 0; i < n; i++)
    powers_aux[i] = NTL::PrepMulModPrecon(rep(powers[i]), p);

  long k = NTL::NextPowerOfTwo(2 * n - 1);
  long k2 = 1L << k; // k2 = 2^k

  Rb.SetSize(k);
  RbInVec.SetLength(k2);

  init_gpu_ntt(k2);

  NTL::zz_pX b(NTL::INIT_SIZE, k2);

  if (NEW_BLUE && n == e) {
    NTL::zz_p rInv = inv(root);
    for (long i = 0; i < n; i++) {
      long iSqr = NTL::MulMod(i, i, e); // i^2 mod 2n
      NTL::zz_p bi = power(rInv, iSqr);
      SetCoeff(b, i, bi);
    }
  } else {
    NTL::zz_p rInv = inv(root);
    SetCoeff(b, n - 1, one); // b[n-1] = 1
    for (long i = 1; i < n; i++) {
      long iSqr = NTL::MulMod(i, i, e); // i^2 mod 2n
      NTL::zz_p bi = power(rInv, iSqr);
      // b[n-1+i] = b[n-1-i] = root^{-i^2}
      SetCoeff(b, n - 1 + i, bi);
      SetCoeff(b, n - 1 - i, bi);
    }
  }

  //Ardhi: get nth-root of unity for n=2^k and current modulus
  NTL::zz_p rtp;
  unsigned int k2= 1L << k; //k2 = 2^k
  FindPrimitiveRoot(rtp, k2); // NTL routine, relative to current modulus
  if (rtp == 0)              // sanity check
    throw RuntimeError("Cmod::compRoots(): no 2^k'th roots of unity mod q");
  long root = NTL::rep(rtp);

  //Ardhi: get psi, psi=root^(1/2)
  long psi;
  NTL::ZZ temp = NTL::SqrRootMod(NTL::conv<NTL::ZZ>(root), NTL::conv<NTL::ZZ>(p));
  NTL::conv(psi, temp);

  //Ardhi: get inverse psi
  long inv_psi = NTL::InvMod(psi, p);
        
  TofftRep(Rb, b, k);
  gpu_ntt(RbInVec, k2, b, p,psi, inv_psi, false);

  // RbInVec = b;
  // RbInPoly[0] = rep(b[0]);
  // RbInPoly[0]=1;
}

__extension__ __int128 flooredDivision(__int128 a, long b)
{
    if(a/b > 0)
      return a/b;

    if(a%b == 0)
      return (a/b);
    else
      return (a/b)-1;

}
__extension__ __int128 myMod2(__int128 a,long b)
{
    return a - b * flooredDivision(a, b);
}

void BluesteinFFT(NTL::zz_pX& x,
                  long n,
                  UNUSED const NTL::zz_p& root,
                  const NTL::zz_pX& powers,
                  const NTL::Vec<NTL::mulmod_precon_t>& powers_aux,
                  const NTL::fftRep& Rb, const NTL::zz_pX& RbInVec)
{
  HELIB_TIMER_START;

  if (IsZero(x))
    return;
  if (n <= 0) {
    clear(x);
    return;
  }

  long p = NTL::zz_p::modulus();

  // std::cout<<"x: "<<x;
  long dx = deg(x);
  for (long i = 0; i <= dx; i++) {
    x[i].LoopHole() =
        NTL::MulModPrecon(rep(x[i]), rep(powers[i]), p, powers_aux[i]);
  }
  x.normalize();

  // std::cout<<"x after MulModPrecon: "<<x<<std::endl;

  long k = NTL::NextPowerOfTwo(2 * n - 1);
  NTL::fftRep& Ra = Cmodulus::getScratch_fftRep(k);
  // Careful! we are multiplying polys of degrees 2*(n-1)
  // and (n-1) modulo x^k-1.  This gives us some
  // truncation in certain cases.

  // std::cout<<"\nmodulus: "<<p<<"\nOmega 2nth root: "<<root<<"\nN: "<<n<<"\nk: "<<k<<std::endl;

  //Ardhi: preparing parameters for gpu ntt
  //Ardhi: get nth-root of unity for n=2^k and current modulus
  NTL::zz_p rtp;
  unsigned int k2= 1L << k; //k2 = 2^k
  FindPrimitiveRoot(rtp, k2); // NTL routine, relative to current modulus
  if (rtp == 0)              // sanity check
    throw RuntimeError("Cmod::compRoots(): no 2^k'th roots of unity mod q");
  long root = NTL::rep(rtp);

  //Ardhi: get psi, psi=root^(1/2)
  long psi;
  NTL::ZZ temp = NTL::SqrRootMod(NTL::conv<NTL::ZZ>(root), NTL::conv<NTL::ZZ>(p));
  NTL::conv(psi, temp);

  //Ardhi: get inverse psi
  long inv_psi = NTL::InvMod(psi, p);

  //Ardhi:
  NTL::vec_zz_p RaInVec(NTL::INIT_SIZE, k2);

  if (NEW_BLUE && n % 2 != 0) {
#if 1
    TofftRep_trunc(Ra, x, k, 2 * n - 1);    //Ardhi: I want to replace this with GPU ntt invocation but the result should be just a vector of long instead of fftRep in NTL
    
    mul(Ra, Ra, Rb); // multiply in FFT representation

    FromfftRep(x, Ra, 0, 2 * (n - 1)); // then convert back
#endif
#if 1
    gpu_ntt(RaInVec, k2, x, p,psi, inv_psi, false); //ForwardFFT

    for (size_t i = 0; i < RaInVec.length(); i++)
    {   
      RaInVec[i] *= RbInVec[i];
    }
    
    gpu_ntt(RaInVec, k2, RaInVec,p, psi, inv_psi,true);//BackwardFFT

  #if 1//check correctness
    for(long i=0; i<= dx; i++){
      std::cout<<"cpu: "<<rep(x[i])<<"gpu: "<<rep(RaInVec[i+n-1])<<std::endl;
    }
  #endif

#endif

    dx = deg(x);
    if (dx >= n) {
      // reduce mod x^n-1
      for (long i = n; i <= dx; i++) {
        x[i - n].LoopHole() = NTL::AddMod(rep(x[i - n]), rep(x[i]), p);
      }
      x.SetLength(n);
      x.normalize();
      dx = deg(x);
    }

    for (long i = 0; i <= dx; i++) {
      x[i].LoopHole() =
          NTL::MulModPrecon(rep(x[i]), rep(powers[i]), p, powers_aux[i]);
    }
    x.normalize();
  } else {
    //Ardhi: I want to replace below code with a call to GPU NTT//But I think I need to verify if I can replace this with the GPU call
    TofftRep_trunc(Ra, x, k, 3 * (n - 1) + 1);     //Requirement: Input x[0...2^k]->x[0...n...2^k], Output x_ntt[2^k]// I don't know why we need parameter (3*(n-1)+1)
    // std::cout<<"RbInPoly :"<<RbInPoly<<std::endl;

    // unsigned int qbit = ceil(std::log2(p));

    // //Ardhi: copy x into buffer a // I am not sure this is needed or not, or we can just use *a =x.rep.data() since vector guaranteed to be contigously allocated
    // for(int i=0; i <= dx; i++)
    //   a[i] = NTL::rep(x.rep[i]);
    // unsigned long long *ntt_a = new unsigned long long[k2];
    // unsigned long long *ntt_b = new unsigned long long[k2];
    //Ardhi: call GPU NTT here
    // gpu_ntt(ntt_a, k2, x, p, psi, inv_psi, false);

    // //Ardhi: check the forward and backward transform
    // gpu_ntt(ntt_b, k2, ntt_a, p, psi, inv_psi, true); 
    // for(long i=0; i<= dx; i++)
    //   std::cout<<"cpu: "<<rep(x[i])<<"gpu: "<<ntt_b[i]<<std::endl;

    // gpu_ntt(ntt_b, k2, RbInPoly, p, psi, inv_psi, false);

    // __extension__ unsigned __int128 buff;

    // for(long i=0; i<k2; i++){
    //   buff = ntt_a[i];
    //   buff = (buff * ntt_b[i]);
    //   ntt_a[i] = myMod2(buff, p);
    // }
    // gpu_ntt(ntt_a, k2, ntt_a, p, psi, inv_psi, true);

    mul(Ra, Ra, Rb); // multiply in FFT representation //Requirement: Input Ra[2^k],Rb[2^k], Output Ra[2^k]

    FromfftRep(x, Ra, n - 1, 2 * (n - 1)); // then convert back //Requirement: Input Ra[2^k], Output x[n]// I don't know why we need parameter (2*(n-1))
    // std::cout<<"x after ntt-mul-intt: "<<x<<std::endl;

    // for(long i=0; i<= dx; i++){
      // std::cout<<"cpu: "<<rep(x[i])<<"gpu: "<<ntt_a[i+n-1]<<std::endl;
    // }

    dx = deg(x);
    for (long i = 0; i <= dx; i++) {
      x[i].LoopHole() =
          NTL::MulModPrecon(rep(x[i]), rep(powers[i]), p, powers_aux[i]);
    }
    x.normalize();
    // std::cout<<"x after ntt-mul-intt-mulmodprecon: "<<x<<std::endl;

  }
}

} // namespace helib
