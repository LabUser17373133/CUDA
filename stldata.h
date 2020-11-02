#ifndef STLDATA_H
#define STLDATA_H
#include "stlnorm.h"
#include "stlvert.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

//stl文件中的数据
class stlData
{
public:
    stlData();
    void readSTL(const std:: string fname);
    stlNorm* m_norm;
    stlVert* m_vert;
    int N_face;

private:

};

#endif // STLDATA_H
