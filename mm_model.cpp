#include <iostream> 
using namespace std;

using std::max;

#define maxTOPS 14.0
#define memBW 650


double calcComputeTime(long int nM, long int nN, long int nK) {
  double numOps = nM * nN * nK;
  return numOps/(maxTOPS*1000000.0);
}

double calcMemTime(long int nM, long int nN, long int nK) {
  double memRead = (nM * nN + nN * nK + nM * nK)/268435456.0;
  return (memRead/memBW) * 1000000;
}

void printTime(long int nM, long int nN, long int nK) {
  double computeTime = calcComputeTime(nM, nN, nK);
  double memTime = calcMemTime(nM, nN, nK);
  cout << nM << "\t" << nN << "\t" << nK << "\t" << "\t" << computeTime << "\t" << memTime << "\t" << max(computeTime, memTime) << "\t" << (nM * nN * nK)/(1000000*max(computeTime, memTime)) << endl; 
}

int main() {
  cout << "m\tn\tk\tcompute time (usec)\tmem time (usec)\tpredicted time\npredicted tflops" << endl;
  printTime(25088, 1, 4096);
  printTime(25088, 2, 4096);
  printTime(25088, 3, 4096);
  printTime(25088, 4, 4096);
  printTime(25088, 5, 4096);
  printTime(25088, 6, 4096);
  printTime(25088, 7, 4096);
  printTime(25088, 8, 4096);
  printTime(25088, 9, 4096);
  printTime(25088, 10, 4096);
  printTime(25088, 16, 4096);
  printTime(25088, 32, 4096);
  printTime(25088, 64, 4096);
  printTime(25088, 128, 4096);
  printTime(25088, 256, 4096);
  printTime(25088, 512, 4096);
  printTime(4096, 1, 1024);
  printTime(4096, 2, 1024);
  printTime(4096, 3, 1024);
  printTime(4096, 4, 1024);
  printTime(4096, 5, 1024);
  printTime(4096, 6, 1024);
  printTime(4096, 7, 1024);
  printTime(4096, 8, 1024);
  printTime(4096, 9, 1024);
  printTime(4096, 10, 1024);
  printTime(4096, 16, 1024);
  printTime(4096, 32, 1024);
  printTime(4096, 64, 1024);
  printTime(4096, 128, 1024);
  printTime(4096, 256, 1024);
  printTime(4096, 512, 1024);
}

