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

//GPU�̷߳���
#define threadsPerBlock 256
#define BlockPergrid 16384

// ���β���
int N = 1024;               // ������
int N_TX = 4096;			// ���������ܲ�������               
int N_RX = 4096;			// ���������ܲ�������

double* Targets;            // Ŀ������ļ�������;6��һ��, 3����������x, y, z(m), 3��������������vx, vy, vz(m)
double* Transmitters_P;     // ����������������, 3��һ��x,y,z(m)
double* Receivers_P;        // ����������������, 3��һ��x,y,z(m)
int* Weights;				// ���Ȩ��
double Angle = 15;			// �Ƕ���ֵ(��)

// ���߲��� 
double AT_Height = 2;       // ���ߵĸ߶�(m)
double AT_Width = 1.5;      // ���ߵĿ��(m)
double AT_Orig_x = -0.75;   // �������в������(m)
double AT_Orig_y = 0;
double AT_Orig_z = 2 * sqrt(2.0);


// STL����ת���� 
// tar Ŀ����������; dat stl������������; N_f ������         
void data2Target(double* tar, stlData dat, int N_f)
{
	for (int i = 0; i < N_f; i++) {
		double* tar_ptr = &tar[i * 6];
		stlVert* v_p = &dat.m_vert[i * 3];
		double c_x, c_y, c_z;       // ���ĵ�����(m)
		c_x = (v_p[0].x + v_p[1].x + v_p[2].x) / 3 / 10;    // ���������겢����
		c_y = (v_p[0].y + v_p[1].y + v_p[2].y) / 3 / 10;
		c_z = (v_p[0].z + v_p[1].z + v_p[2].z) / 3 / 10;
		*tar_ptr = c_x;
		*(tar_ptr + 1) = c_y;
		*(tar_ptr + 2) = c_z;
		*(tar_ptr + 3) = dat.m_norm[i].x;     // ��������
		*(tar_ptr + 4) = dat.m_norm[i].y;
		*(tar_ptr + 5) = dat.m_norm[i].z;
	}
}

// ����λ�������ʼ�� 
// p ���ߵ�����; N_s ��������; w, h ��͸�(m); o_() ԭ������(m)
void initAntenna(double* p, int N_s, double w, double h, double o_x, double o_y, double o_z)
{
	double x, y, z, dx, dy, dz;
	x = o_x; y = o_y; z = o_z;			// ����ԭ������
	int N_w = (int)(sqrt((double)N_s)); // ����Ĳ�������
	int N_h = (int)(sqrt((double)N_s));
	dx = w / N_w;						// �������
	dy = h / N_h;
	dz = 0;
	for (int i = 0; i < N_s; i++) {   // ƽ��������
		double* p_ptr = &p[i * 3];
		int m = i % N_w;
		int n = i / N_h;
		*p_ptr = x + m * dx;
		*(p_ptr + 1) = y + n * dy;
		*(p_ptr + 2) = z + dz;
	}
}

// �жϽ������ߵ�λ���Ƿ�����ֵ�Ƕ���;
// �Ƿ���1,���Ƿ���0
__device__ int is_Reflect(double* Target, double* Transmitter, double* Receiver, double Threshold)      //Ŀ��� ����� ���ܵ� ��ֵ(cos15)
{
	double Incident_Light[3];								//�������
	Incident_Light[0] = Target[0] - Transmitter[0];
	Incident_Light[1] = Target[1] - Transmitter[1];
	Incident_Light[2] = Target[2] - Transmitter[2];

	//�ж��Ƿ�����
	double temp1 = Incident_Light[0] * Target[3] + Incident_Light[1] * Target[4] + Incident_Light[2] * Target[5];          // T*N
	if (temp1 > 0)
		return 0;

	//���㷴�����
	double temp2 = pow(Target[3], 2) + pow(Target[4], 2) + pow(Target[5], 2);          //����ģ��ƽ��

	double Reflected_Light[3];							//�������
	Reflected_Light[0] = -2 * temp1 * Target[3] + Incident_Light[0] * temp2;
	Reflected_Light[1] = -2 * temp1 * Target[4] + Incident_Light[1] * temp2;
	Reflected_Light[2] = -2 * temp1 * Target[5] + Incident_Light[2] * temp2;

	//�жϷ���Ƕ��Ƿ�����
	double Received_Light[3];							//���չ���
	Received_Light[0] = Receiver[0] - Target[0];
	Received_Light[1] = Receiver[1] - Target[1];
	Received_Light[2] = Receiver[2] - Target[2];

	double temp3 = Reflected_Light[0] * Received_Light[0] + Reflected_Light[1] * Received_Light[1] + Reflected_Light[2] * Received_Light[2];        //R1*R2
	double temp4 = sqrt(pow(Reflected_Light[0], 2) + pow(Reflected_Light[1], 2) + pow(Reflected_Light[2], 2)) * sqrt(pow(Received_Light[0], 2) + pow(Received_Light[1], 2) + pow(Received_Light[2], 2)); //|R1|*|R2|

	if (temp3 > temp4 * Threshold)			//�Ƕ�С����ֵ�Ƕ�
		return 1;
	else
		return 0;
}


// �������(Ȩ��)�ļ���
__global__ void Illumination_Area_Calculation(double* Targets, double* Transmitters, double* Receivers, int* Weights, double* Angle, int* Num_Targets, int* Num_Transmitters, int* Num_Receivers)
{
	double Threshold = cos(*Angle / 180 * M_PI);						//��ֵ cos(�Ƕ�)

	int Index_Targets = blockIdx.x;										//Ŀ������
	int Index_Transmitters_Receives_Pair = threadIdx.x;
	int Index_Transmitters_Receives_Pair_0 = threadIdx.x;				//����������߶Ե����

	__shared__ int Weights_cache[threadsPerBlock];				//������

	while (Index_Targets < *Num_Targets) {				//Ŀ��ѭ��
		double* Target_Position = &Targets[Index_Targets * 6];						//Ŀ���λ��

		int temp = 0;
		while (Index_Transmitters_Receives_Pair < (*Num_Transmitters) * (*Num_Receivers)) {			//���߶�ѭ��
			int j = Index_Transmitters_Receives_Pair / (*Num_Transmitters);               //�ڼ�����������
			int i = Index_Transmitters_Receives_Pair - j * (*Num_Transmitters);           //�ڼ�����������
			double* Transmitter_Position = &Transmitters[i * 3];				//���߶�λ��
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

		//������һ��Ŀ���ѭ��
		Index_Transmitters_Receives_Pair = Index_Transmitters_Receives_Pair_0;
		Index_Targets = Index_Targets + gridDim.x;
	}
}


//************************************//
void Exposure_Weights(double* Targets, double* Transmitters, double* Receivers, int* Weights, double* Angle, int* Num_Targets, int* Num_Transmitters, int* Num_Receivers)
{
	//�豸�еı�������
	double* dev_Targets, * dev_Transmitters, * dev_Receivers, * dev_Angle;
	int* dev_Weights, * dev_Num_Targets, * dev_Num_Transmitters, * dev_Num_Receivers;

	//Ϊ�豸�����ݿ��ٿռ�
	cudaMalloc((void**)& dev_Targets, *Num_Targets * 6 * sizeof(double));
	cudaMalloc((void**)& dev_Transmitters, *Num_Transmitters * 3 * sizeof(double));
	cudaMalloc((void**)& dev_Receivers, *Num_Receivers * 3 * sizeof(double));
	cudaMalloc((void**)& dev_Angle, sizeof(double));
	cudaMalloc((void**)& dev_Weights, *Num_Targets * sizeof(int));
	cudaMalloc((void**)& dev_Num_Targets, sizeof(int));
	cudaMalloc((void**)& dev_Num_Transmitters, sizeof(int));
	cudaMalloc((void**)& dev_Num_Receivers, sizeof(int));

	//��������
	cudaMemcpy(dev_Targets, Targets, *Num_Targets * 6 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Transmitters, Transmitters, *Num_Transmitters * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Receivers, Receivers, *Num_Receivers * 3 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Angle, Angle, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Weights, Weights, *Num_Targets * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Num_Targets, Num_Targets, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Num_Transmitters, Num_Transmitters, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Num_Receivers, Num_Receivers, sizeof(int), cudaMemcpyHostToDevice);

	//ʹ��CUDA����Ȩ������
	Illumination_Area_Calculation << <BlockPergrid, threadsPerBlock >> > (dev_Targets, dev_Transmitters, dev_Receivers, dev_Weights, dev_Angle, dev_Num_Targets, dev_Num_Transmitters, dev_Num_Receivers);

	//���豸�е�Ȩ�����ݿ�������
	cudaMemcpy(Weights, dev_Weights, *Num_Targets * sizeof(int), cudaMemcpyDeviceToHost);

	//�ͷ��ڴ�ռ�
	cudaFree(dev_Targets);
	cudaFree(dev_Transmitters);
	cudaFree(dev_Receivers);
	cudaFree(dev_Angle);
	cudaFree(dev_Weights);
	cudaFree(dev_Num_Targets);
	cudaFree(dev_Num_Transmitters);
	cudaFree(dev_Num_Receivers);

	//������д���ļ�����
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
	stlData m_stl;                                      // STL�ļ����ݵ���
	m_stl.readSTL("man05.stl");							// ��STL�ļ�
	N = m_stl.N_face;                                   // ��ȡ������
	Targets = (double*)malloc(sizeof(double) * N * 6);
	data2Target(Targets, m_stl, N);                     // STL����ת����һά����
	Weights = (int*)malloc(sizeof(int) * N);
	std::fill(Weights, Weights + N, 0);                 // Ȩ������  
	Transmitters_P = (double*)malloc(sizeof(double) * N_TX * 3);
	Receivers_P = (double*)malloc(sizeof(double) * N_RX * 3);

	//��ʼ����������
	initAntenna(Transmitters_P, N_TX, AT_Width, AT_Height, AT_Orig_x, AT_Orig_y, AT_Orig_z);
	initAntenna(Receivers_P, N_RX, AT_Width, AT_Height, AT_Orig_x, AT_Orig_y, AT_Orig_z);

	Exposure_Weights(Targets, Transmitters_P, Receivers_P, Weights, &Angle, &N, &N_TX, &N_RX);

	for (int i = 0; i < N; i++) {
		printf("%d %d\n", i, Weights[i]);
	}
	return 0;
}



//************* ��֤���� ***************//

// ������������
//for (int i = 0; i < N_TX; i++)
//{
//	double* ptr = &Transmitters_P[i * 3];
//	printf("%f  %f  %f\n", ptr[0], ptr[1], ptr[2]);
//}

// ������������
//for (int i = 0; i < N_TX; i++)
//{
//	double* ptr = &Receivers_P[i * 3];
//	printf("%f  %f  %f\n", ptr[0], ptr[1], ptr[2]);
//}