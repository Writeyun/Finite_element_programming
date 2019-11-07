# -*- coding: utf-8 -*-

#******************************************************************************
"""
                       **********程序说明 **********
    本程序为有限元小编程，其用途仅于求解三角形三节点常应变单元在结点荷载作用下的应
力、应变问题。在编写过程整体思路来源龙驭球先生的《有限元概论》。本程序分为三大模块，
分别为输入模块、计算模块和输出模块，具体模块的划分可见源码内部注释。结合有限元求解
过程，程序实现步骤为离散化、单元分析、整体分析、引入支撑条件、求解方程组以及回代求
解目标量。
    1.离散化过程为使用者在输入的文件中自行定义参数内容；
    
    2.单元分析，目的是为得到单元刚度和其他的单元参数，通过定义类Element_charas实现
该目的，通过在类内部定义多个函数求取其单元特性，这样有利于后期代码调试和维护。当然
可以选择更好的方式以达到计算效率的提升，本处未采用；

    3.整体分析，根据"对号入座"的原则，从单元刚度集成总刚。通过函数global_stiffness
实现整体刚度的集成，本处未采用通用的方法——对单元刚度中的子单元当都依次取出，再集装
入总体刚度。
     本程序利用python中的特有数据结构DataFram形式，通过设置标签——方向编号，对单元
刚度和整体刚度均采用设置标签的方式，通过标签直接对应，贴合“对号入座”的认识，同时规
避了对应过程中差“1”的小麻烦。另外在集成整体刚度过程中，对单元刚度采用“要一个就调用
一个，用完即扔”的原则；当然，缺点是未利用整体刚度矩阵的对称行性和稀疏性对整体刚度采
用“半带存储”的方式；

    4.引入支撑条件，通过“化零置一法”，对未采用半带存储的整体刚度矩阵，由支撑所约束
方向的编号对应整体刚度的主元素化为1，而在整体刚度矩阵中，该元素所在的行和列其他元素
全部置换成0，对于荷载矩阵同样需要对该方向对应的荷载元素置换为0.对于半带存储整体矩阵
，由于位置的变换，需要寻找变换后的元素位置，进行变换。实现原理和具体可参见相关书籍；

    5.求解方程组，定义函数LU_decom，itera_solu，采用“LU分解法”，分别完成消元和迭
代的步骤，输出结点位移。

    6.回代求解目标量，利用弹性力学知识，定义函数，由结点位移，反求结点力，由结点力
求出主应力和主应力方向
    
    *.本程序总流程是计算参数输入——>单元几何特性——>应变矩阵——>弹性矩阵——>应力矩阵——
>单元刚度——>整体刚度——>引入边界条件——>求解线性方程组——>由结点位移求结点力——>....

    *本程序在实现过程中，关于有限元原理以及程序实现内容，参考了龙驭球先生和王勖成先
的著作成果，在此声明!!!


                     **********参数说明 **********
1.question_type : It is include just two choicecs, one is 1, the other is 0, 
    if the number enqual 1,whice means plane strain problem,when it is 0, whice 
    means plane stress problem.
    
2.base_number : It is a list include number of nodes,numbers of elements numbe-
    rs of displacements,numbers of struts,half bandwidth,numbers of node load.
    
3.material_paras：It include the material's characteristic,such as elastic 
    modulus,Poisson's ratio,thickness and bulk weight. 
    
4.joints_corrds: It is the number of joints' coords.

5.ele_joint_numbers means element joints numbers.

6.strut_array means the array of strut.

7.load_array is the array of nodes' loads,the first number is tne load direc-
    tion, the section is the load value.    
"""
#******************************************************************************

import numpy as np
import pandas as pd


#******************************************************************************
#输入模块
#定义函数 def read_parameters,实现从指定文件读取计算所需参数的功能
def read_parameters(filename):
    
    elements_paras = {}
    calcul_parameters, paras_names = [],[]

    with open(filename) as file_object:
        elements_paras = eval(file_object.read()) 
        for key, value in elements_paras.items():
            paras_names.append(key)
            calcul_parameters.append(np.array(value))
    
    return calcul_parameters


#******************************************************************************
#计算模块
    
#定义类Element_charas，计算指定三角形单元特性
class Element_charas():
    
    def __init__(self,ele_nub,joints_nubs,joints_cods,elastic,
                                              poisson, thickness):
        """参数初始化"""
        self.ele_nub = ele_nub
        self.joints_nubs = joints_nubs
        self.joints_coords = joints_coords
        self.elastic = elastic
        self.poisson = poisson
        self.thickness = thickness
        self.earea = 0
        self.abcx = []
        self.node_coords = []
        self.elest_mat = []
        self.strain_mat = []
        self.stress_mat = []
        self.stiff_mat = []
        
    def coord_paras(self):
        """计算单元的坐标特性"""
        joint_list = list(self.joints_nubs[self.ele_nub])
        self.node_coords = self.joints_coords[joint_list]
        nd = self.node_coords
        b1 = nd[1,1]-nd[2,1]
        b2 = nd[2,1]-nd[0,1]
        b3 = nd[0,1]-nd[1,1]
        c1 = nd[2,0]-nd[1,0]
        c2 = nd[0,0]-nd[2,0]
        c3 = nd[1,0]-nd[0,0]
        self.abcx = np.array([[b1,b2,b3],[c1,c2,c3]])
                 
    def element_area(self):
        """计算单元的面积"""
        a_matrix = np.c_[np.ones((3,1)),self.node_coords]
        self.earea = 0.5 * np.linalg.det(a_matrix)
        
        return self.earea
    
    def strain_matrix(self):
        """单元应变矩阵"""
        abc = self.abcx
        strain_mat = np.array([[abc[0,0], 0, abc[0,1], 0, abc[0,2],0],
                [0, abc[1,0], 0, abc[1,1], 0, abc[1,2]],
                [abc[1,0], abc[0,0], abc[1,1], abc[0,1], abc[1,2], abc[0,2]]])
        self.strain_mat = strain_mat /(2 * self.earea)
        
        return self.strain_mat
    
    
    def elestic_matrix(self):
        """生成弹性矩阵"""
        ele, minu = self.elastic, self.poisson
        para1 = ele / (1 - minu**2)
        elest_mat = np.array([[1, minu, 0],
                            [minu, 1, 0],
                            [0, 0, (1-minu)*0.5]])
        self.elest_mat =para1 * elest_mat
        
        return self.elest_mat
    
    
    def stress_matrix(self):
        """生成应力矩阵"""
        self.stress_mat = np.matmul(self.elest_mat, self.strain_mat)
        
        return self.stress_mat
        
    def element_stiffness(self):
        """生成单元刚度"""
        stiff_mat = np.dot(self.strain_mat.T, self.stress_mat)
        self.stiff_mat = stiff_mat * self.thickness * self.earea 
        
        return self.stiff_mat
#完成类定义  
         
#******************************************************************************

#初始化整体刚度
def init_globke(n):
    data2 = np.zeros((n*2, n*2))
    col_ind =list(range(1,n*2+1))
    glob_ke = pd.DataFrame(data2, columns=col_ind, index=col_ind)
    
    return glob_ke


#定义函数生成整体刚度      
def global_stiffness(nubs,eke,gke):
    """集成整体刚度矩阵"""
    dir_vector = []
    for jnb in nubs:
        dir_vector.append(jnb*2-1)
        dir_vector.append(jnb*2)
        
    ekes = pd.DataFrame(eke,columns=dir_vector, index=dir_vector)
    
    for j in dir_vector:
        for i in dir_vector:
            gke[i][j] = gke[i][j] + ekes[i][j]
            
    return gke
            

#定义函数，生成荷载向量
#荷载元素i对应的方向上的荷载，大小为loadment的第一列对应的值
def load_vectorp(loadnbs,displace,loadment):
    """本函数生成荷载矩阵"""
    loadp = np.zeros((displace,1))#初始化荷载向量
    
    if loadnbs > 0:
        for i in range(1, loadnbs +1):
            j = int(loadment[i][1] - 1)
            loadp[j] = loadment[i][0]  
            
    return loadp
            
    
#定义函数，根据‘化零置一法’，修改整体刚度，荷载向量，实现支撑条件的引入
def support_gke(strut_para,gke_para):
    """本函数对整体刚度矩阵引入支撑条件"""
    for n in strut_para:
        gke_para.loc[n] = 0 #将该行更改为零
        gke_para.loc[:, n] = 0 #将该列更改为零
        gke_para.loc[n, n] = 1 #将该主元素置为1  
        
    return gke_para 
 
       
def support_loadp(strut_para,load_para):
    """本函数对荷载向量引入支撑条件"""
    for n in strut_para:
        load_para[n-1] = 0 #将对应的荷载列修改为零
        
    return load_para



def LU_decom(A):
    """本函数用于对系数方程进行LU分解"""
    n = len(A[0])
    for i in range(n):
        if i ==0:
            for j in range(1,n):
                A[j][0] = A[j][0]/A[0][0]
        else:
            for j in range(i,n):
                temp1 = 0
                for k in range(0, i):
                    temp1 = temp1 + A[i][k] * A[k][j]
                A[i][j] = A[i][j] - temp1
            for j in range(i+1, n):
                temp2 = 0
                for k in range(0, i):
                    temp2 = temp2 + A[j][k] * A[k][i]
                A[j][i] = (A[j][i] - temp2)/A[i][i]
                
    return A
    
    
 
def itera_solu(A,B):
    """本函数用于迭代求解LU分解后的线性方程组"""
    n = len(A[0])
    y = np.zeros((n,1))
    x = np.zeros((n,1))
    
    y[0] = B[0]
    for i in range(1,n):
        temp3 = 0
        for k in range(0,i):
            temp3 = temp3 + A[i][k] * y[k]
        y[i] = B[i] - temp3
      
    x[n-1] = y[n-1] / A[n-1][n-1]
    for i in range(n-2,-1,-1):
        temp4 = 0
        for k in range(n-1,0,-1):
            temp4 = temp4 + A[i][k] * x[k]
        x[i] = y[i] - temp4
#        print("y:{}, x:{}, temp:{}".format(y[i], x[i], temp4))
    
    return x
        
        
def displacement_print(mv):
    """本函数用于格式化输出节点位移"""
    print('\n\n%22s******** NODES DISPLACEMENTS ********'%(' '))
    print("\n%20sJD=%12sU=%20sV="%(' ',' ',' '))
    for i in range(0,base_numbers[0]):
        print("\n%20s%-3i%20.10f%20.10f"%('',i+1, mv[i][0], mv[i][1]))
    

def nodemv_reshape(base_nbs,joints_nbs,mv_para):
    """将节点位移矩阵（3*2）转换成单元位移向量（1*6）"""
    e_mv = []
    for e in range(1, base_nbs+1):
        joints_numb = joints_nbs[e]
        for i in joints_numb:
            e_mv.append(list(mv_para[i-1]))
    e_mv = np.array(e_mv).reshape((-1,6))

    return e_mv
  
def element_stress(base_nbs, strain_para, emv_para):
    """由应变矩阵和单元位移向量求单元应力Sx,Sy,Tau"""
    strain_list = []
    for e in range(base_nbs):
        strain_list.append(strain_para.dot(emv_para[e].T))
    return strain_list

def stress_print(enubs,stre):
    """本函数格式化打印单元应力"""
    print('\n\n%25s>>>>>>> ELEMENT  STRESS <<<<<<<'%(' '))
    for e in range(enubs):
        print("\n%18sELE_NUMB%2s%-5i"%('','',e+1))
        print("%18sSx=%-14.7f Sy=%-14.7f tau=%-14.7f"%(
                ' ',stre[e][0],stre[e][1],stre[e][2]))

    
#******************************************************************************
    
#提取单元的计算参数   
calcul_paras  =  read_parameters('elements_paramenters.txt')#读取文件内计算参数
base_numbers = calcul_paras[0]
#基本参数：结点个数、单元个数、位移分量、支杆个数、半带宽和结点荷载个数      
question_type = calcul_paras[1]#问题类型：0-平面应力问题；1-平面应变问题
mater_paras = calcul_paras[2]#材料特性：弹性模量、泊松比、厚度和容重
joints_coords = calcul_paras[3].reshape(-1,2)#结点坐标数组
joints_numbers = calcul_paras[4].reshape(-1,3)#单元结点编码
strut_array = calcul_paras[5]#结构整体支撑数组
load_array = calcul_paras[6].reshape(-1, 2)#结点荷载数组


#******************************************************************************


glob_ke = init_globke(base_numbers[0])#初始化整体刚度，格式为DataFrame格式
for e in range(1, base_numbers[1]+1):
    #遍历单元循环，计算每个单元刚度，最后集装生成总体刚度
    now_element = Element_charas(e, joints_numbers, joints_coords,
                mater_paras[0],mater_paras[1],mater_paras[2] ) #创建实例
    now_element.coord_paras() #计算单元几何特性
    earea = now_element.element_area() #计算单元面积
    strain_mat = now_element.strain_matrix() #计算应变矩阵
    elest_mat = now_element.elestic_matrix() #计算弹性矩阵
    stress_mat = now_element.stress_matrix() #计算应力矩阵
    stiff_mat = now_element.element_stiffness() #计算单元刚度矩阵
    gkes = global_stiffness(joints_numbers[e],stiff_mat,glob_ke) #集装整体刚度


loadps = load_vectorp(base_numbers[5],base_numbers[2],load_array) #生成荷载向量
gkes_support = support_gke(strut_array,gkes) #整体刚度引入支撑条件
loadps_support = support_loadp(strut_array,loadps) #荷载向量引入支撑条件

KE = gkes_support.values #将gkes_support中的值提取出来,此步骤降低更新负影响
LUKE = LU_decom(KE) #LU分解整体刚度矩阵
node_mv = itera_solu(LUKE,loadps_support).reshape((-1,2))#迭代
print_mv = displacement_print(node_mv) #格式化输出结点位移
eleme_shape = nodemv_reshape(base_numbers[1],joints_numbers,node_mv)#单元位移
eleme_stress = np.array(element_stress(base_numbers[1],
                     stress_mat, eleme_shape)).reshape((-1,3)) #单元应力
print_stress = stress_print(base_numbers[1],eleme_stress)#打印单元应力













