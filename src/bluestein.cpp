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
#include <NTL/FFT_impl.h>

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
                   NTL::fftRep& Rb, NTL::vec_zz_p& RbInVec, const NTL::zz_p& psi, NTL::zz_pX& RbInPoly, std::vector<unsigned long long>& gpu_powers)
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
  // myPsi.SetLength(1);

  init_gpu_ntt(k2);

  NTL::zz_pX b(NTL::INIT_SIZE, k2);

  //Ardhi: I think below codes prepare for the the chirp signal
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

  RbInPoly = b;

#if 0
  TofftRep(Rb, b, k);
#endif
#if 1
  
  long inv_psi = NTL::InvMod(rep(psi), p);


  // gpu_ntt(RbInVec, k2, b, p,rep(psi), NTL::InvMod(rep(psi), p), false); //Ardhi: convert b->RbInVec aka vec<long>//zz_pX to vec_zz_p
  gpu_ntt_forward(RbInVec, k2, b, p, gpu_powers, rep(psi), inv_psi); //Ardhi: convert b->RbInVec aka vec<long>//zz_pX to vec_zz_p
#endif
#if 0 //check the forward transform is correct or not
  NTL::vec_zz_p reverse_RbInVec;
  reverse_RbInVec.SetLength(k2);
  gpu_ntt(reverse_RbInVec, k2, RbInVec, p, rep(psi), inv_psi, true);
  // std::cout<<"\npsi: "<<rep(psi)<<" RbInVec: ";
  long dx = deg(b);
  for (long i = 0; i <= dx; i++)
  {
    // std::cout<<RbInVec[i]<<" ";
    if(reverse_RbInVec[i] != b.rep[i]){
          printf("b: %lu, reverseRbInVec: %lu\n", rep(b.rep[i]), rep(reverse_RbInVec[i]));
          throw RuntimeError("Cmod::bluesteinInit(): b to RbInVec conversion error");
    }
  }
#endif

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
                  const NTL::fftRep& Rb, const NTL::vec_zz_p& RbInVec, const NTL::zz_p& psi, const NTL::zz_pX& RbInPoly, const std::vector<unsigned long long>& gpu_powers, UNUSED const std::vector<unsigned long long>& gpu_ipowers)
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
  // NTL::zz_p rtp;
  unsigned int k2= 1L << k; //k2 = 2^k
  // FindPrimitiveRoot(rtp, k2); // NTL routine, relative to current modulus
  // if (rtp == 0)              // sanity check
  //   throw RuntimeError("Cmod::compRoots(): no 2^k'th roots of unity mod q");
  // long k2_root = NTL::rep(rtp);

  //Ardhi: get psi, psi=root^(1/2)
  // long psi;
  // NTL::ZZ temp = NTL::SqrRootMod(NTL::conv<NTL::ZZ>(k2_root), NTL::conv<NTL::ZZ>(p));
  // NTL::conv(psi, temp);

  //Ardhi: check psi and myPsi
  // if(psi != myPsi[0])
  //   throw RuntimeError("Cmod::bluesteinFFT(): psi and myPsi missmatch");


  // //Ardhi: get inverse psi
  long inv_psi = NTL::InvMod(rep(psi), p);

  //Ardhi:
  NTL::vec_zz_p RaInVec(NTL::INIT_SIZE, k2);

  if (NEW_BLUE && n % 2 != 0) {
#if 0
    TofftRep_trunc(Ra, x, k, 2 * n - 1);    //Ardhi: I want to replace this with GPU ntt invocation but the result should be just a vector of long instead of fftRep
    mul(Ra, Ra, Rb); // multiply in FFT representation
    
#endif
#if 0 //compare ntt of RbInPoly with RbInVec
  NTL::vec_zz_p test(NTL::INIT_SIZE, k2);
  gpu_ntt(test, k2, RbInPoly, p,rep(psi), inv_psi, false); //ForwardFFT
  for (long i = 0; i < RbInVec.length(); i++)
  {
    if(test[i] != RbInVec[i])
          throw RuntimeError("Cmod::bluesteinFFT(): RbInVec does not match test");
  }

  gpu_ntt(test, k2, RbInVec, p,rep(psi), inv_psi, true); //BackwardFFT
  long count = RbInPoly.rep.length();
  for (long i = 0; i < count; i++)
  {
    if(test[i] != RbInPoly.rep[i])
          throw RuntimeError("Cmod::bluesteinFFT(): RbInVec does not match test");
  }
#endif
#if 1 //Ardhi: Naive BluesteinGPU
  // gpu_ntt(RaInVec, k2, x, p,rep(psi), inv_psi, false); //ForwardFFT // zz_pX to vec_zz_p
  gpu_ntt_forward(RaInVec, k2, x, p, gpu_powers, rep(psi), inv_psi); //ForwardFFT // zz_pX to vec_zz_p

  // long iteration = NTL::FFTRoundUp(2 * n -1, k); // The NTL implementation only performing mult. up to len=FFTRoundUp of the NTT
  for (long i = 0; i < RaInVec.length(); i++)
  {   
    RaInVec[i] *= RbInVec[i];
  }
  
  // gpu_ntt(RaInVec, k2, RaInVec,p, rep(psi), inv_psi,true);//BackwardFFT //vec_zz_p to vec_zz_p
  gpu_ntt_backward(RaInVec, k2, RaInVec,p, gpu_ipowers, rep(psi), inv_psi);//BackwardFFT //vec_zz_p to vec_zz_p
#else
  gpu_fused_polymul(RaInVec, k2, x, p, gpu_powers, gpu_ipowers, rep(psi), inv_psi);
#endif

#if 1
    long l = 2*(n-1)+1;
    x.rep.SetLength(l);
    for (long i = 0; i < l; i++) {
      x[i].LoopHole() = rep(RaInVec[i]);
    }
    x.normalize();
#endif
#if 0
    FromfftRep(x, Ra, 0, 2 * (n - 1)); // then convert back
    #if 0//check correctness
      dx = deg(x);
      for(long i=0; i<= dx; i++){
        if(rep(x[i]) != rep(RaInVec[i]))
              throw RuntimeError("Cmod::bluesteinFFT(): gpu conv and cpu conv is missmatch");
      }
      
      NTL::zz_pX temp;
      long l = 2*(n-1)+1;
      temp.rep.SetLength(l);
      for (long i = 0; i < l; i++) {
        temp[i].LoopHole() = rep(RaInVec[i]);
      }
      temp.normalize();

      long temp_d = deg(temp);
      if(temp_d != dx){
              printf("temp_d: %lu, dx: %lu\n", temp_d, dx);
              throw RuntimeError("Cmod::bluesteinFFT(): temp_d and dx is missmatch");       
      }
      // for(long i=0; i<= dx; i++){
      //   std::cout<<"cpu: "<<rep(x[i])<<std::endl;
      // }

      // for(long i=0; i< RaInVec.length(); i++){
      //   std::cout<<"gpu: "<<rep(RaInVec[i])<<std::endl;
      // }
    #endif
#endif
    dx = deg(x);
    // printf("dx:%lu \n", dx);
    if (dx >= n) {
      // reduce mod x^n-1
      for (long i = n; i <= dx; i++) {
        #if 1
          x[i - n].LoopHole() = NTL::AddMod(rep(x[i - n]), rep(x[i]), p);
        #else
          x[i - n].LoopHole() = NTL::AddMod(rep(RaInVec[i - n]), rep(RaInVec[i]), p);
        #endif
      }
      x.SetLength(n);
      x.normalize();
      dx = deg(x);
    }

#if 0 //check x after modulo
    for(long i=0; i<= dx; i++){
      std::cout<<"cpu after mod x^n-1: "<<rep(x[i])<<std::endl;
    }
#endif

    for (long i = 0; i <= dx; i++) {
      x[i].LoopHole() =
          NTL::MulModPrecon(rep(x[i]), rep(powers[i]), p, powers_aux[i]);
    }
    x.normalize();
  } else {

    //Ardhi: this section of code related to gpu ntt needs to be fixed
    //Ardhi: I want to replace below code with a call to GPU NTT//But I think I need to verify if I can replace this with the GPU call
    TofftRep_trunc(Ra, x, k, 3 * (n - 1) + 1);     //Requirement: Input x[0...2^k]->x[0...n...2^k], Output x_ntt[2^k]// I don't know why we need parameter (3*(n-1)+1)
    // std::cout<<"RbInPoly :"<<RbInPoly<<std::endl;

    // unsigned int qbit = ceil(std::log2(p));

#if 1 //check with older code
    // //Ardhi: copy x into buffer a // I am not sure this is needed or not, or we can just use *a =x.rep.data() since vector guaranteed to be contigously allocated
    // for(int i=0; i <= dx; i++)
    //   a[i] = NTL::rep(x.rep[i]);
    unsigned long long *ntt_a = new unsigned long long[k2];
    unsigned long long *ntt_b = new unsigned long long[k2];
    //Ardhi: call GPU NTT here
    gpu_ntt(ntt_a, k2, x, p, rep(psi), inv_psi, false);

    // //Ardhi: check the forward and backward transform
    // gpu_ntt(ntt_b, k2, ntt_a, p, psi, inv_psi, true); 
    // for(long i=0; i<= dx; i++)
    //   std::cout<<"cpu: "<<rep(x[i])<<"gpu: "<<ntt_b[i]<<std::endl;

    gpu_ntt(ntt_b, k2, RbInPoly, p, rep(psi), inv_psi, false);

    #if 1 //compare ntt_b with RbInVec
      for (long i = 0; i < RbInVec.length(); i++)
      {
        if(ntt_b[i] != RbInVec[i])
              throw RuntimeError("Cmod::bluesteinFFT(): RbInVec does not match ntt_b");
      }
    #endif

    __extension__ unsigned __int128 buff;

    for(long i=0; i<k2; i++){
      buff = ntt_a[i];
      buff = (buff * ntt_b[i]);
      ntt_a[i] = myMod2(buff, p);
    }
    gpu_ntt(ntt_a, k2, ntt_a, p, rep(psi), inv_psi, true);
#endif
    mul(Ra, Ra, Rb); // multiply in FFT representation //Requirement: Input Ra[2^k],Rb[2^k], Output Ra[2^k]

#if 1
    gpu_ntt(RaInVec, k2, x, p, rep(psi), inv_psi, false); //ForwardFFT

    #if 1 //check forward gpu ntt for RaInVec
      NTL::vec_zz_p reverse_Ra(NTL::INIT_SIZE, k2);
      gpu_ntt(reverse_Ra, k2, RaInVec, p, rep(psi), inv_psi, true);
      dx = deg(x);
      for (long i = 0; i <= dx; i++)
      {
        if(reverse_Ra[i] != x.rep[i])
              throw RuntimeError("Cmod::bluesteinFFT(): x to RaInVec conversion error");
      }
    #endif

    for (long i = 0; i < RaInVec.length(); i++)
    { 
      #define check_mulmod 1  
      #if check_mulmod //check mulmod correctness
         __extension__ unsigned __int128 buff;
        buff = rep(RaInVec[i]);
      #endif

      RaInVec[i] *= RbInVec[i];

      #if check_mulmod //check mulmod correctness
        buff = (buff * rep(RbInVec[i]));
        if(rep(RaInVec[i]) != myMod2(buff, p))
          throw RuntimeError("Cmod::bluesteinFFT(): mulmod error");
      #endif
    }

    gpu_ntt(RaInVec, k2, RaInVec, p, rep(psi), inv_psi,true);//BackwardFFT
#endif

    FromfftRep(x, Ra, n - 1, 2 * (n - 1)); // then convert back //Requirement: Input Ra[2^k], Output x[n]// I don't know why we need parameter (2*(n-1))
    // std::cout<<"x after ntt-mul-intt: "<<x<<std::endl;
    
  #if 1//check correctness
    dx = deg(x);
    // for(long i=0; i<= dx; i++){
    //   std::cout<<"cpu: "<<rep(x[i])<<" gpu: "<<ntt_a[i+n-1]<<std::endl;
    // }
    
    for(long i=0; i<= dx; i++){
      std::cout<<"cpu: "<<rep(x[i])<<std::endl;
    }

    for(long i=0; i< RaInVec.length(); i++){
      std::cout<<"gpu1: "<<rep(RaInVec[i])<<std::endl;
    }

    for(long i=0; i< k2; i++){
      std::cout<<"gpu2: "<<ntt_a[i]<<std::endl;
    }    
  #endif

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
