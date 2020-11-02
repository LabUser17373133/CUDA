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
#include "stlvert.h"
#include "stldata.h"
#include "stlnorm.h"

// 几何参数
int N = 1024;               // 总面数
int NT = 256;               // 发射天线采样点               
int NR = 256;               // 接收天线采样点

double* Targets;            // 目标物体的几何数据;6个一组, 3个面心坐标x, y, z(m), 3个法线向量坐标vx, vy, vz(m)
double* Transmitters_P;     // 发射天线坐标数据, 3个一组x,y,z(m)
double* Receivers_P;        // 接收天线坐标数据, 3个一组x,y,z(m)
double* Weights;            // 面的权重
double Angle = 15;          // 角度阈值(rad)

// 天线参数 
double AT_Height = 2;       // 天线的高度(m)
double AT_Width = 1;        // 天线的宽度(m)
double AT_Orig_x = -0.5;    // 天线阵列采样起点(m)
double AT_Orig_y = 0;
double AT_Orig_z = 2;


 
// STL数据转数组 
// tar 目标物体数组; dat stl读到的数据类; N_f 总面数         
void data2Target(double* tar, stlData dat, int N_f)
{
    for (int i = 0; i < N_f; i++) {
        double* tar_p = &tar[i * 6];  
        stlVert* v_p = &dat.m_vert[i * 3];    
        double c_x, c_y, c_z;       // 中心点坐标m
        c_x = (v_p[0].x + v_p[1].x + v_p[2].x) / 3 / 10;    // 求中心坐标并缩放
        c_y = (v_p[0].y + v_p[1].y + v_p[2].y) / 3 / 10;
        c_z = (v_p[0].z + v_p[1].z + v_p[2].z) / 3 / 10;
        *tar_p = c_x;
        *(tar_p + 1) = c_y;
        *(tar_p + 2) = c_z;
        *(tar_p + 3) = dat.m_norm[i].x;     // 法线向量
        *(tar_p + 4) = dat.m_norm[i].y;
        *(tar_p + 5) = dat.m_norm[i].z;
    }
}

// 天线初始化 
// p 天线点数组; N_s 采样点数; w, h 宽和高(m); o_() 原点坐标(m)
void initAntenna(double* p, int N_s, double w, double h, double o_x, double o_y, double o_z)
{
    double x, y, z, dx, dy, dz;
    x = o_x; y = o_y; z = o_z;      // 天线原点坐标
    dx = w / N_s;                   // 采样间隔
    dy = h / N_s;
    dz = 0;
    for (int i = 0; i < N_s * N_s; i++) {   // 平面矩阵遍历
        double* p_ptr = &p[i * 3];
        int m = i % N_s;
        int n = i / N_s;
        *p_ptr = x + m * dx;
        *(p_ptr + 1) = y + n * dy;
        *(p_ptr + 2) = z + dz;
    }
}






__device__ int is_Reflect(double* T, double* TX, double* RX, double TH)      //目标点 发射点 接受点 阈值(cos15)
{
    double IL[3];               //入射光线
    IL[0] = T[0] - TX[0];
    IL[1] = T[1] - TX[1];
    IL[2] = T[2] - TX[2];

    //判断是否正射
    double temp1 = IL[0] * T[3] + IL[1] * T[4] + IL[2] * T[5];          // T*N
    if (temp1 > 0)
        return 0;

    //计算反射光线
    double temp2 = pow(T[3], 2) + pow(T[4], 2) + pow(T[5], 2);          //法线模长平方
    double RL[3];               //反射光线
    RL[0] = -2 * temp1 * T[3] + IL[0] * temp2;
    RL[1] = -2 * temp1 * T[4] + IL[1] * temp2;
    RL[2] = -2 * temp1 * T[5] + IL[2] * temp2;
    //printf("%lf %lf %lf\n", RL[0], RL[1], RL[2]);

    //判断反射角度是否满足
    double RL2[3];              //接收光线
    RL2[0] = RX[0] - T[0];
    RL2[1] = RX[1] - T[1];
    RL2[2] = RX[2] - T[2];

    double temp3 = RL[0] * RL2[0] + RL[1] * RL2[1] + RL[2] * RL2[2];        //R1*R2
    double temp4 = sqrt(pow(RL[0], 2) + pow(RL[1], 2) + pow(RL[2], 2)) * sqrt(pow(RL2[0], 2) + pow(RL2[1], 2) + pow(RL2[2], 2)); //|R1|*|R2|
 
    if (temp3 > temp4 * TH)
        return 1;
    else
        return 0;
}

__global__ void Illumination_Area_Calculation(double* Target, double* Weight, double* Transmitter, double* Receiver,double* Angle,int* N,int* NT,int* NR)
{
    int nT = blockIdx.x;            //目标的序号
    int nTR0 = threadIdx.x;          //发射接收天线的序号
    int nTR = threadIdx.x;
    double TH = cos(*Angle / 180 * M_PI);        //阈值 cos(角度)

    while (nT < *N) {
        //printf("%d\n", nT);
        //Weight[nT] = 0;
        double* T = &Target[nT * 6];        //目标的位置
        while (nTR < (*NT) * (*NR)) {
            //printf("%d %d\n", nT,nTR);
            int j = nTR / (*NT);               //第几个接收天线
            int i = nTR - j * (*NT);           //第几个发射天线
            double* TX = &Transmitter[i * 3];
            double* RX = &Receiver[j * 3];
            //printf("%f\n", is_Reflect(T, TX, RX, TH));

            if (is_Reflect(T, TX, RX, TH) == 1)
                Weight[nT]++;

            nTR = nTR + blockDim.x;
        }
        nTR = nTR0;
        nT = nT + gridDim.x;
    }
}





int main()
{
    stlData m_stl;                                      // STL文件数据的类
    m_stl.readSTL("man05.stl");
    N = m_stl.N_face;                                   // 获取总面数
    Targets = (double*)malloc(sizeof(double) * N * 6);
    data2Target(Targets, m_stl, N);                     // STL数据转换到一维数组
    Weights = (double*)malloc(sizeof(double) * N);
    std::fill(Weights, Weights+N, 0);                   // 权重清零  
    Transmitters_P = (double*)malloc(sizeof(double) * NT * NT * 3);     
    Receivers_P = (double*)malloc(sizeof(double) * NR * NR * 3);

    //初始化天线坐标
    initAntenna(Transmitters_P, NT, AT_Width, AT_Height, AT_Orig_x, AT_Orig_y, AT_Orig_z);
    initAntenna(Receivers_P, NR, AT_Width, AT_Height, AT_Orig_x, AT_Orig_y, AT_Orig_z);


    double *dev_Target, *dev_Transmitter, *dev_Receiver;
    double *dev_Angle;
    double* dev_Weight;
    int *dev_N, *dev_NT, *dev_NR;

    //为设备中数据开辟空间
    cudaMalloc((void**)&dev_Target, N * 6 * sizeof(double));
    cudaMalloc((void**)&dev_Transmitter, NT * 3 * sizeof(double));
    cudaMalloc((void**)&dev_Receiver, NR * 3 * sizeof(double));
    cudaMalloc((void**)&dev_Angle, sizeof(double));
    cudaMalloc((void**)&dev_Weight, N * sizeof(double));
    cudaMalloc((void**)&dev_N, sizeof(int));
    cudaMalloc((void**)&dev_NT, sizeof(int));
    cudaMalloc((void**)&dev_NR, sizeof(int));

    //复制数据
    cudaMemcpy(dev_Target, Targets, N * 6 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Transmitter, Transmitters_P, NT * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Receiver, Receivers_P, NR * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Angle, &Angle, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Weight, Weights, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_N, &N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_NT, &NT, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_NR, &NR, sizeof(int), cudaMemcpyHostToDevice);

    Illumination_Area_Calculation <<<60000, 500>>> (dev_Target, dev_Weight, dev_Transmitter, dev_Receiver, dev_Angle, dev_N, dev_NT, dev_NR);

    cudaMemcpy(Weights, dev_Weight, N * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%f\n", Weights[i]);
    }
    return 0;
}