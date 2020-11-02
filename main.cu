
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void VectAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
int main()
{
    printf("Hello World From CPU!\n");
    float A_h[] = { 1, 2, 3 };
    float B_h[] = { 1, 2, 3 };
    float C_h[] = { 3, 2, 1 };
    
    
    float *A_d, *B_d, *C_d;
    cudaMalloc((float**)&A_d, sizeof(A_h));
    cudaMalloc((float**)&B_d, sizeof(B_h));
    cudaMalloc((float**)&C_d, sizeof(C_h));
    cudaMemcpy(A_d, A_h, sizeof(A_h), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(B_h), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, sizeof(C_h), cudaMemcpyHostToDevice);
    VectAdd <<<1, 3 >>> (A_d, B_d, C_d);
    cudaMemcpy(A_h, A_d, sizeof(A_h), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_h, B_d, sizeof(B_h), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_h, C_d, sizeof(C_h), cudaMemcpyDeviceToHost);
    printf("%f, %f, %f\n", C_h[0], C_h[1] , C_h[2]);
    cudaDeviceReset();
    return 0;
}
