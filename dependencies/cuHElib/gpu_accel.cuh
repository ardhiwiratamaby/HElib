#ifndef GPU_ACCEL_H
#define GPU_ACCEL_H


void InitGPUBuffer(long phim, int n_rows);
void DestroyGPUBuffer();

void setMapA(long index, long data);
void setMapB(long index, long data);
void setModulus(long index, long data);

long getMapA(long index);
long getMapB(long index);

void InitContiguousHostMapModulus(long phim, int n_rows);

void CudaEltwiseAddMod();
void CudaEltwiseAddMod(long scalar);
void CudaEltwiseSubMod();
void CudaEltwiseSubMod(long scalar);
void CudaEltwiseMultMod();
void CudaEltwiseMultMod(long scalar);

int cuda_add();

#endif