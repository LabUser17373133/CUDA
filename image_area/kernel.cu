#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "stlvert.h"
#include "stldata.h"
#include "stlnorm.h"

//GPU线程分配
#define threadsPerBlock 256
#define BlockPergrid 16384

// 几何参数
int N = 1024;               // 总面数
int N_TX = 4096;			// 发射天线总采样点数               
int N_RX = 4096;			// 接收天线总采样点数

double* Targets;            // 目标物体的几何数据;6个一组, 3个面心坐标x, y, z(m), 3个法线向量坐标vx, vy, vz(m)
double* Transmitters_P;     // 发射天线坐标数据, 3个一组x,y,z(m)
double* Receivers_P;        // 接收天线坐标数据, 3个一组x,y,z(m)
int* Weights;				// 面的权重
double Angle = 15;			// 角度阈值(度)

// 天线参数 
double AT_Height = 2;       // 天线的高度(m)
double AT_Width = 1.5;      // 天线的宽度(m)
double AT_Orig_x = -0.75;   // 天线阵列采样起点(m)
double AT_Orig_y = 0;
double AT_Orig_z = 2 * sqrt(2.0);


// STL数据转数组 
// tar 目标物体数组; dat stl读到的数据类; N_f 总面数         
void data2Target(double* tar, stlData dat, int N_f)
{
	for (int i = 0; i < N_f; i++) {
		double* tar_ptr = &tar[i * 6];
		stlVert* v_p = &dat.m_vert[i * 3];
		double c_x, c_y, c_z;       // 中心点坐标(m)
		c_x = (v_p[0].x + v_p[1].x + v_p[2].x) / 3 / 10;    // 求中心坐标并缩放
		c_y = (v_p[0].y + v_p[1].y + v_p[2].y) / 3 / 10;
		c_z = (v_p[0].z + v_p[1].z + v_p[2].z) / 3 / 10;
		*tar_ptr = c_x;
		*(tar_ptr + 1) = c_y;
		*(tar_ptr + 2) = c_z;
		*(tar_ptr + 3) = dat.m_norm[i].x;     // 法线向量
		*(tar_ptr + 4) = dat.m_norm[i].y;
		*(tar_ptr + 5) = dat.m_norm[i].z;
	}
}

// 天线位置坐标初始化 
// p 天线点数组; N_s 采样点数; w, h 宽和高(m); o_() 原点坐标(m)
void initAntenna(double* p, int N_s, double w, double h, double o_x, double o_y, double o_z)
{
	double x, y, z, dx, dy, dz;
	x = o_x; y = o_y; z = o_z;			// 天线原点坐标
	int N_w = (int)(sqrt((double)N_s)); // 长宽的采样点数
	int N_h = (int)(sqrt((double)N_s));
	dx = w / N_w;						// 采样间隔
	dy = h / N_h;
	dz = 0;
	for (int i = 0; i < N_s; i++) {   // 平面矩阵遍历
		double* p_ptr = &p[i * 3];
		int m = i % N_w;
		int n = i / N_h;
		*p_ptr = x + m * dx;
		*(p_ptr + 1) = y + n * dy;
		*(p_ptr + 2) = z + dz;
	}
}

// 判断接收天线的位置是否在阈值角度内;
// 是返回1,不是返回0
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

	if (temp3 > temp4 * Threshold)			//角度小于阈值角度
		return 1;
	else
		return 0;
}


// 照射面积(权重)的计算
__global__ void Illumination_Area_Calculation(double* Targets, double* Transmitters, double* Receivers, int* Weights, double* Angle, int* Num_Targets, int* Num_Transmitters, int* Num_Receivers)
{
	double Threshold = cos(*Angle / 180 * M_PI);						//阈值 cos(角度)

	int Index_Targets = blockIdx.x;										//目标的序号
	int Index_Transmitters_Receives_Pair = threadIdx.x;
	int Index_Transmitters_Receives_Pair_0 = threadIdx.x;				//发射接收天线对的序号

	__shared__ int Weights_cache[threadsPerBlock];				//共享缓存

	while (Index_Targets < *Num_Targets) {				//目标循环
		double* Target_Position = &Targets[Index_Targets * 6];						//目标的位置

		int temp = 0;
		while (Index_Transmitters_Receives_Pair < (*Num_Transmitters) * (*Num_Receivers)) {			//天线对循环
			int j = Index_Transmitters_Receives_Pair / (*Num_Transmitters);               //第几个接收天线
			int i = Index_Transmitters_Receives_Pair - j * (*Num_Transmitters);           //第几个发射天线
			double* Transmitter_Position = &Transmitters[i * 3];				//天线对位置
			double* Receiver_Position = &Receivers[j * 3];

			if (is_Reflect(Target_Position, Transmitter_Position, Receiver_Position, Threshold) == 1)
				temp++;

			Index_Transmitters_Receives_Pair = Index_Transmitters_Receives_Pair + blockDim.x;
		}
		Weights_cache[threadIdx.x] = temp;

		__syncthreads();

		int k = blockDim.x / 2;
		while (k != 0) {
			if (threadIdx.x < k)
				Weights_cache[threadIdx.x] = Weights_cache[threadIdx.x] + Weights_cache[threadIdx.x + k];
			__syncthreads();
			k = k / 2;
		}

		if (threadIdx.x == 0)
			Weights[Index_Targets] = Weights_cache[0];

		//进行下一个目标的循环
		Index_Transmitters_Receives_Pair = Index_Transmitters_Receives_Pair_0;
		Index_Targets = Index_Targets + gridDim.x;
	}
}


//************************************//
void Exposure_Weights(double* Targets, double* Transmitters, double* Receivers, int* Weights, double* Angle, int* Num_Targets, int* Num_Transmitters, int* Num_Receivers)
{
	//设备中的变量定义
	double* dev_Targets, * dev_Transmitters, * dev_Receivers, * dev_Angle;
	int* dev_Weights, * dev_Num_Targets, * dev_Num_Transmitters, * dev_Num_Receivers;

	//为设备中数据开辟空间
	cudaMalloc((void**)& dev_Targets, *Num_Targets * 6 * sizeof(double));
	cudaMalloc((void**)& dev_Transmitters, *Num_Transmitters * 3 * sizeof(double));
	cudaMalloc((void**)& dev_Receivers, *Num_Receivers * 3 * sizeof(double));
	cudaMalloc((void**)& dev_Angle, sizeof(double));
	cudaMalloc((void**)& dev_Weights, *Num_Targets * sizeof(int));
	cudaMalloc((void**)& dev_Num_Targets, sizeof(int));
	cudaMalloc((void**)& dev_Num_Transmitters, sizeof(int));
	cudaMalloc((void**)& dev_Num_Receivers, sizeof(int));

	//复制数据
	cudaMemcpy(dev_Targets, Targets, *Num_Targets * 6 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Transmitters, Transmitters, *Num_Transmitters * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Receivers, Receivers, *Num_Receivers * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Angle, Angle, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Weights, Weights, *Num_Targets * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Num_Targets, Num_Targets, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Num_Transmitters, Num_Transmitters, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Num_Receivers, Num_Receivers, sizeof(int), cudaMemcpyHostToDevice);

	//使用CUDA计算权重数据
	Illumination_Area_Calculation << <BlockPergrid, threadsPerBlock >> > (dev_Targets, dev_Transmitters, dev_Receivers, dev_Weights, dev_Angle, dev_Num_Targets, dev_Num_Transmitters, dev_Num_Receivers);

	//将设备中的权重数据拷贝回来
	cudaMemcpy(Weights, dev_Weights, *Num_Targets * sizeof(int), cudaMemcpyDeviceToHost);

	//释放内存空间
	cudaFree(dev_Targets);
	cudaFree(dev_Transmitters);
	cudaFree(dev_Receivers);
	cudaFree(dev_Angle);
	cudaFree(dev_Weights);
	cudaFree(dev_Num_Targets);
	cudaFree(dev_Num_Transmitters);
	cudaFree(dev_Num_Receivers);

	//将数据写进文件里面
	FILE* fid_Weights;
	fid_Weights = fopen("Weights.txt", "w");

	for (int i = 0; i < *Num_Targets; i++)
		fprintf(fid_Weights, "%d\n", Weights[i]);

	fclose(fid_Weights);
}


//***********************************
//********** Main Code **************
//***********************************
int main()
{
	stlData m_stl;                                      // STL文件数据的类
	m_stl.readSTL("man05.stl");							// 读STL文件
	N = m_stl.N_face;                                   // 获取总面数
	Targets = (double*)malloc(sizeof(double) * N * 6);
	data2Target(Targets, m_stl, N);                     // STL数据转换到一维数组
	Weights = (int*)malloc(sizeof(int) * N);
	std::fill(Weights, Weights + N, 0);                 // 权重清零  
	Transmitters_P = (double*)malloc(sizeof(double) * N_TX * 3);
	Receivers_P = (double*)malloc(sizeof(double) * N_RX * 3);

	//初始化天线坐标
	initAntenna(Transmitters_P, N_TX, AT_Width, AT_Height, AT_Orig_x, AT_Orig_y, AT_Orig_z);
	initAntenna(Receivers_P, N_RX, AT_Width, AT_Height, AT_Orig_x, AT_Orig_y, AT_Orig_z);

	Exposure_Weights(Targets, Transmitters_P, Receivers_P, Weights, &Angle, &N, &N_TX, &N_RX);

	for (int i = 0; i < N; i++) {
		printf("%d %d\n", i, Weights[i]);
	}
	return 0;
}



//************* 验证数据 ***************//

// 发射天线坐标
//for (int i = 0; i < N_TX; i++)
//{
//	double* ptr = &Transmitters_P[i * 3];
//	printf("%f  %f  %f\n", ptr[0], ptr[1], ptr[2]);
//}

// 接收天线坐标
//for (int i = 0; i < N_TX; i++)
//{
//	double* ptr = &Receivers_P[i * 3];
//	printf("%f  %f  %f\n", ptr[0], ptr[1], ptr[2]);
//}