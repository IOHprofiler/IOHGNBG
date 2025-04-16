/// Author: Vladimir Stanovov (vladimirstanovov@yandex.ru)
/// Last edited: March 7th, 2024
/// C++ implementation of Generalized Numerical Benchmark Generator (GNBG) instance for GECCO 2024 competition
/// Includes implementation of simple Differential Evolution (DE) with rand/1 strategy and binomial crossover
/// Problem parameters are read from f#.txt files which should be prepared with python script convert.py from f#.mat
/// Competition page: https://competition-hub.github.io/GNBG-Competition/
/// Reference:
/// D. Yazdani, M. N. Omidvar, D. Yazdani, K. Deb, and A. H. Gandomi, "GNBG: A Generalized
///   and Configurable Benchmark Generator for Continuous Numerical Optimization," arXiv prepring	arXiv:2312.07083, 2023.
/// A. H. Gandomi, D. Yazdani, M. N. Omidvar, and K. Deb, "GNBG-Generated Test Suite for Box-Constrained Numerical Global
///   Optimization," arXiv preprint arXiv:2312.07034, 2023.
/// MATLAB version: https://github.com/Danial-Yazdani/GNBG_Generator.MATLAB
/// Python version: https://github.com/Danial-Yazdani/GNBG_Generator.Python
#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>
#include <random>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

using namespace std;
unsigned globalseed = unsigned(time(NULL));//2024;//
std::mt19937 generator_uni_i(globalseed);
std::mt19937 generator_uni_r(globalseed+100);
std::uniform_int_distribution<int> uni_int(0,32768);
std::uniform_real_distribution<double> uni_real(0.0,1.0);


int IntRandom(int target) {if(target == 0) return 0; return uni_int(generator_uni_i)%target;}
double Random(double minimal, double maximal) {return uni_real(generator_uni_r)*(maximal-minimal)+minimal;}

static std::string ROOT = "";

class GNBG
{
public:
    int FEval;
    int MaxEvals;
    int Dimension;
    int CompNum;
    double MinCoordinate;
    double MaxCoordinate;
    double AcceptanceThreshold;
    double OptimumValue;
    double BestFoundResult;
    double fval;
    double AcceptanceReachPoint;
    double* CompSigma;
    double* Lambda;
    std::vector<double> OptimumPosition;
    double* FEhistory;
    double* temp;
    double* a;
    double** CompMinPos;
    double** CompH;
    double** Mu;
    double** Omega;
    double*** RotationMatrix;
    GNBG(int func_num);
    ~GNBG();
    double Fitness(const std::vector<double>& xvec);

};

GNBG::GNBG(int func_num): OptimumPosition(30)
{
    char buffer[15];
    sprintf(buffer, "f%d.txt", func_num);
    const std::string path = (ROOT + std::string(buffer));
    
    ifstream fin(path);
    AcceptanceReachPoint = -1;
    FEval = 0;
    fin>>MaxEvals;
    fin>>AcceptanceThreshold;
    fin>>Dimension;
    fin>>CompNum;
    fin>>MinCoordinate;
    fin>>MaxCoordinate;
    FEhistory = new double[MaxEvals];
    a = new double[Dimension];
    temp = new double[Dimension];
    CompSigma = new double[CompNum];
    Lambda = new double[CompNum];
    CompMinPos = new double*[CompNum];
    CompH = new double*[CompNum];
    Mu = new double*[CompNum];
    Omega = new double*[CompNum];
    RotationMatrix = new double**[CompNum];
    for(int i=0;i!=CompNum;i++)
    {
        CompMinPos[i] = new double[Dimension];
        CompH[i] = new double[Dimension];
        Mu[i] = new double[2];
        Omega[i] = new double[4];
        RotationMatrix[i] = new double*[Dimension];
        for(int j=0;j!=Dimension;j++)
            RotationMatrix[i][j] = new double[Dimension];
    }
    for(int i=0;i!=CompNum;i++)
        for(int j=0;j!=Dimension;j++)
            fin>>CompMinPos[i][j];
    for(int i=0;i!=CompNum;i++)
        fin>>CompSigma[i];
    for(int i=0;i!=CompNum;i++)
        for(int j=0;j!=Dimension;j++)
            fin>>CompH[i][j];
    for(int i=0;i!=CompNum;i++)
        for(int j=0;j!=2;j++)
            fin>>Mu[i][j];
    for(int i=0;i!=CompNum;i++)
        for(int j=0;j!=4;j++)
            fin>>Omega[i][j];
    for(int i=0;i!=CompNum;i++)
        fin>>Lambda[i];
    for(int j=0;j!=Dimension;j++)
        for(int k=0;k!=Dimension;k++)
            for(int i=0;i!=CompNum;i++)
                fin>>RotationMatrix[i][j][k];
    fin>>OptimumValue;
    for(int i=0;i!=Dimension;i++)
    {
        fin>>OptimumPosition[i];
    }

}
double GNBG::Fitness(const std::vector<double>& xvec)
{
    double res = 0;
    for(int i=0;i!=CompNum;i++)
    {
        for(int j=0;j!=Dimension;j++)
            a[j] = xvec[j] - CompMinPos[i][j];
        for(int j=0;j!=Dimension;j++)
        {
            temp[j] = 0;
            for(int k=0;k!=Dimension;k++)
                temp[j] += RotationMatrix[i][j][k]*a[k]; //matmul rotation matrix and (x - peak position)
        }
        for(int j=0;j!=Dimension;j++)
        {
            if(temp[j] > 0)
                a[j] = exp(log( temp[j])+Mu[i][0]*(sin(Omega[i][0]*log( temp[j]))+sin(Omega[i][1]*log( temp[j]))));
            else if(temp[j] < 0)
                a[j] =-exp(log(-temp[j])+Mu[i][1]*(sin(Omega[i][2]*log(-temp[j]))+sin(Omega[i][3]*log(-temp[j]))));
            else
                a[j] = 0;
        }
        fval = 0;
        for(int j=0;j!=Dimension;j++)
            fval += a[j]*a[j]*CompH[i][j];
        fval = CompSigma[i] + pow(fval,Lambda[i]);
        res = (i == 0)*fval + (i != 0)*min(res,fval);//if first iter then save fval, else take min
    }
    if(FEval > MaxEvals)
        return res;
    FEhistory[FEval] = res;
    BestFoundResult = (FEval == 0)*res + (FEval != 0)*min(res,BestFoundResult);
    if(FEhistory[FEval] - OptimumValue < AcceptanceThreshold && AcceptanceReachPoint == -1)
       AcceptanceReachPoint = FEval;
    FEval++;
    return res;
}
GNBG::~GNBG()
{
    delete a;
    delete temp;
    delete CompSigma;
    delete Lambda;
    for(int i=0;i!=CompNum;i++)
    {
        delete CompMinPos[i];
        delete CompH[i];
        delete Mu[i];
        delete Omega[i];
        for(int j=0;j!=Dimension;j++)
            delete RotationMatrix[i][j];
        delete RotationMatrix[i];
    }
    delete CompMinPos;
    delete CompH;
    delete Mu;
    delete Omega;
    delete RotationMatrix;
    delete FEhistory;
}


void set_root(const std::string& path)
{
    ROOT = path;
}


PYBIND11_MODULE(gnbgcpp, m)
{
    m.def("set_root", &set_root, py::arg("path"));
    
    py::class_<GNBG>(m, "GNBG")
        .def(py::init<int>(), py::arg("fid"))
        .def("__call__", &GNBG::Fitness, py::arg("x"))
        .def_readonly("FEval", &GNBG::FEval)
        .def_readonly("MaxEvals", &GNBG::MaxEvals)
        .def_readonly("Dimension", &GNBG::Dimension)
        .def_readonly("CompNum", &GNBG::CompNum)
        .def_readonly("MinCoordinate", &GNBG::MinCoordinate)
        .def_readonly("MaxCoordinate", &GNBG::MaxCoordinate)
        .def_readonly("AcceptanceThreshold", &GNBG::AcceptanceThreshold)
        .def_readonly("OptimumValue", &GNBG::OptimumValue)
        .def_readonly("BestFoundResult", &GNBG::BestFoundResult)
        .def_readonly("fval", &GNBG::fval)
        .def_readonly("AcceptanceReachPoint", &GNBG::AcceptanceReachPoint)
        // .def_readonly("CompSigma", &GNBG:CompSigma)
        // .def_readonly("Lambda", &GNBG:Lambda)
        .def_readonly("OptimumPosition", &GNBG::OptimumPosition)
        // .def_readonly("FEhistory", &GNBG:FEhistory)
        // .def_readonly("temp", &GNBG:temp)
        // .def_readonly("a", &GNBG:a)
        // .def_readonly("CompMinPos", &GNBG:CompMinPos)
        // .def_readonly("CompH", &GNBG:CompH)
        // .def_readonly("Mu", &GNBG:Mu)
        // .def_readonly("Omega", &GNBG:Omega)
        // .def_readonly("RotationMatrix", &GNBG:RotationMatrix)
    ;
}


// int main()
// {
//     cout.precision(20);
//     int NRuns = 31;
//     double* Errors = new double[NRuns];
//     double* AcceptancePoints = new double[NRuns];
//     for(int func_num=1;func_num!=25;func_num++)
//     {
//         int NNonEmpty = 0;
//         double meanAcceptance = 0;
//         double meanError = 0;
//         double stdAcceptance = 0;
//         double stdError = 0;
//         for(int run=0;run!=NRuns;run++)
//         {
//             cout<<"Func: "<<func_num<<"\tRun: "<<run<<"\t";
//             GNBG gnbg(func_num);
//             Optimizer Opt(/*population size*/ 100, gnbg);
//             Opt.Run(gnbg);
//             if(true) // save for graphs?
//             {
//                 char buffer[100];
//                 sprintf(buffer,"Res_DE_f%d_r%d.txt",func_num,run);
//                 ofstream fout_c(buffer);
//                 fout_c.precision(20);
//                 for(int i=0;i!=Opt.BestHistoryIndex;i++)
//                     fout_c<<Opt.BestHistory[i]-gnbg.OptimumValue<<"\n";
//                 fout_c.close();
//             }
//             Errors[run] = gnbg.BestFoundResult - gnbg.OptimumValue;
//             AcceptancePoints[run] = gnbg.AcceptanceReachPoint;
//             meanError += Errors[run];
//             meanAcceptance += AcceptancePoints[run]*(AcceptancePoints[run] != -1);
//             NNonEmpty += (AcceptancePoints[run] != -1);
//             cout<<"Error: "<<Errors[run]<<"\tAcceptancePoint: "<<AcceptancePoints[run]<<endl;
//         }
//         meanError /= double(NRuns);
//         if(NNonEmpty > 0)
//             meanAcceptance = meanAcceptance / double(NNonEmpty);
//         else
//             meanAcceptance = 0;
//         for(int run=0;run!=NRuns;run++)
//         {
//             stdError += (Errors[run]-meanError)*(Errors[run]-meanError);
//             stdAcceptance += (AcceptancePoints[run]-meanAcceptance)*(AcceptancePoints[run]-meanAcceptance)*(AcceptancePoints[run] != -1);
//         }
//         if(NRuns > 1)
//             stdError = sqrt(stdError / double(NRuns-1));
//         else
//             stdError = 0;
//         if(NNonEmpty > 1)
//             stdAcceptance = sqrt(stdAcceptance / double(NNonEmpty-1));
//         else
//             stdAcceptance = 0;
//         cout<<"Average FE to reach acceptance result: "<<meanAcceptance<<"\t("<<stdAcceptance<<")\n";
//         cout<<"Acceptance Ratio: "<<double(NNonEmpty)/double(NRuns)*100<<"\n";
//         cout<<"Final result: "<<meanError<<"\t("<<stdError<<")\n";
//     }
//     delete Errors;
//     delete AcceptancePoints;
//     return 0;
// }
