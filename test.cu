#define _USE_MATH_DEFINES
#include <math.h>
#include <complex>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define imin(a, b) (a < b ? a : b)

const int threadPerBlock


__global__ void dot(float *a, float *b, float *c)
{
    __share__ float c_cache[]

}