#ifndef GPU_ACCEL_H
#define GPU_ACCEL_H


void InitGPUBuffer(long phim);
void DestroyGPUBuffer();
void CudaEltwiseAddMod(long* result, const long* a, const long* b, long size, long modulus);
void CudaEltwiseAddMod(long* result, const long* a, long scalar, long size, long modulus);
void CudaEltwiseSubMod(long* result, const long* a, const long* b, long size, long modulus);
void CudaEltwiseSubMod(long* result, const long* a, long scalar, long size, long modulus);
void CudaEltwiseMultMod(long* result, const long* a, const long* b, long size, long modulus);
void CudaEltwiseMultMod(long* result, const long* a, long scalar, long size, long modulus);

int cuda_add();

#endif