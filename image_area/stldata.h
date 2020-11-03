#ifndef STLDATA_H
#define STLDATA_H
#include "stlnorm.h"
#include "stlvert.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

// stl文件中的数据
class stlData
{
public:
    stlData();
    void readSTL(const std:: string fname);
    stlNorm* m_norm;    // 法线向量
    stlVert* m_vert;    // 点坐标(文件顺序)
    int N_face;         // 总面数

private:

};


#endif // STLDATA_H
