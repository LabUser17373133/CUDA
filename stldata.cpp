#include "stldata.h"


stlData::stlData()
{

}

/*读取stl(二进制)*/
void stlData:: readSTL(const std:: string  fname)
{
    char name[80];                                  //文件头

    std::ifstream in(fname, std::ifstream::in | std::ifstream::binary);
    if (!in) {
        std::cout << "FILE READ ERROR.\n" << std::endl;
    }
    in.read(name, 80);
    in.read((char*)&N_face, sizeof(int));
    m_norm = (stlNorm*)malloc(sizeof(stlNorm) * N_face);
    m_vert = (stlVert*)malloc(sizeof(stlVert) * N_face * 3);
    int v_i = 0;                                        //第v_i个点
    for(int i = 0; i < N_face; i++){
        float coorXYZ[12];
        in.read((char*)coorXYZ, 12 * sizeof (float));
        m_norm[i].setXYZ(coorXYZ[0], coorXYZ[1], coorXYZ[2]);
        m_vert[v_i++].setXYZ(coorXYZ[3], coorXYZ[4], coorXYZ[5]);
        m_vert[v_i++].setXYZ(coorXYZ[6], coorXYZ[7], coorXYZ[8]);
        m_vert[v_i++].setXYZ(coorXYZ[9], coorXYZ[10], coorXYZ[11]);
        in.read((char*)coorXYZ, 2);                     //跳过颜色
    }
}


