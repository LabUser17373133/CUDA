#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__device__ int is_Reflect(double* Target, double* Transmitter, double* Receiver, double Threshold)      //目标点 发射点 接受点 阈值(cos15)
{
	double Incident_Light[3];								//入射光线
	Incident_Light[0] = Target[0] - Transmitter[0];
	Incident_Light[1] = Target[1] - Transmitter[1];
	Incident_Light[2] = Target[2] - Transmitter[2];

	//判断是否正射
	double temp1 = Incident_Light[0] * Target[3] + Incident_Light[1] * Target[4] + Incident_Light[2] * Target[5];          // T*N
	if (temp1 > 0)
		return 0;

	//计算反射光线
	double temp2 = pow(Target[3], 2) + pow(Target[4], 2) + pow(Target[5], 2);          //法线模长平方

	double Reflected_Light[3];							//反射光线
	Reflected_Light[0] = -2 * temp1 * Target[3] + Incident_Light[0] * temp2;
	Reflected_Light[1] = -2 * temp1 * Target[4] + Incident_Light[1] * temp2;
	Reflected_Light[2] = -2 * temp1 * Target[5] + Incident_Light[2] * temp2;

	//判断反射角度是否满足
	double Received_Light[3];							//接收光线
	Received_Light[0] = Receiver[0] - Target[0];
	Received_Light[1] = Receiver[1] - Target[1];
	Received_Light[2] = Receiver[2] - Target[2];

	double temp3 = Reflected_Light[0] * Received_Light[0] + Reflected_Light[1] * Received_Light[1] + Reflected_Light[2] * Received_Light[2];        //R1*R2
	double temp4 = sqrt(pow(Reflected_Light[0], 2) + pow(Reflected_Light[1], 2) + pow(Reflected_Light[2], 2)) * sqrt(pow(Received_Light[0], 2) + pow(Received_Light[1], 2) + pow(Received_Light[2], 2)); //|R1|*|R2|

	if (temp3 > temp4* Threshold)
		return 1;
	else
		return 0;
}

__global__ void Illumination_Area_Calculation(double* Targets, double* Transmitters, double* Receivers, int* Weights, double* Angle, int* Num_Targets, int* Num_Transmitters, int* Num_Receivers)
{
	int num_Targets = blockIdx.x;								//目标的序号
	int num_Transmitters_Receives_Pair_0 = threadIdx.x;			//发射接收天线对的序号
	int num_Transmittewrs_Receives_Pair = threadIdx.x;
	double TH = cos(*Angle / 180 * M_PI);        //阈值 cos(角度)

	while (num_Targets < *Num_Targets) {
		//printf("%d\n", nT);
		//Weight[nT] = 0;
		double* T = &Targets[num_Targets * 6];        //目标的位置
		while (num_Transmitters_Receives_Pair < (*Num_Transmitters) * (*Num_Receivers)) {
			//printf("%d %d\n", nT,nTR);
			int j = num_Transmitters_Receives_Pair / (*Num_Transmitters);               //第几个接收天线
			int i = num_Transmitters_Receives_Pair - j * (*Num_Transmitters);           //第几个发射天线
			double* TX = &Transmitters[i * 3];
			double* RX = &Receivers[j * 3];
			//printf("%f\n", is_Reflect(T, TX, RX, TH));

			if (is_Reflect(T, TX, RX, TH) == 1)
				Weights[num_Targets]++;

			num_Transmitters_Receives_Pair = num_Transmitters_Receives_Pair + blockDim.x;
		}
		num_Transmitters_Receives_Pair = num_Transmitters_Receives_Pair_0;
		num_Targets = num_Targets + gridDim.x;
	}
}

void Exposure_Weights(double* Targets, double* Transmitters, double* Receivers, int* Weights, double* Angle, int* Num_Targets, int* Num_Transmitters, int* Num_Receivers) 
{
	//设备中的变量定义
	double* dev_Targets, * dev_Transmitters, * dev_Receivers, * dev_Angle;
	int* dev_Weights, * dev_Num_Targets, * dev_Num_Transmitters, * dev_Num_Receivers;

	//为设备中数据开辟空间
	cudaMalloc((void**)&dev_Targets, *Num_Targets * 6 * sizeof(double));
	cudaMalloc((void**)&dev_Transmitters, *Num_Transmitters * 3 * sizeof(double));
	cudaMalloc((void**)&dev_Receivers, *Num_Receivers * 3 * sizeof(double));
	cudaMalloc((void**)&dev_Angle, sizeof(double));
	cudaMalloc((void**)&dev_Weights, *Num_Targets * sizeof(int));
	cudaMalloc((void**)&dev_Num_Targets, sizeof(int));
	cudaMalloc((void**)&dev_Num_Transmitters, sizeof(int));
	cudaMalloc((void**)&dev_Num_Receivers, sizeof(int));

	//复制数据
	cudaMemcpy(dev_Targets, Targets, *Num_Targets * 6 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Transmitters, Transmitters, *Num_Transmitters * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Receivers, Receivers, *Num_Receivers * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Angle, Angle, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Num_Targets, Num_Targets, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Num_Transmitters, Num_Transmitters, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Num_Receivers, &Num_Receivers, sizeof(int), cudaMemcpyHostToDevice);

	Illumination_Area_Calculation <<<32768, 256 >>> (dev_Target, dev_Weight, dev_Transmitter, dev_Receiver, dev_Angle, dev_N, dev_NT, dev_NR);

	cudaMemcpy(Weights, dev_Weights, *Num_Targets * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_Targets);
	cudaFree(dev_Transmitters);
	cudaFree(dev_Receivers);
	cudaFree(dev_Angle);
	cudaFree(dev_Weights);
	cudaFree(dev_Num_Targets);
	cudaFree(dev_Num_Transmitters);
	cudaFree(dev_Num_Receivers);
}