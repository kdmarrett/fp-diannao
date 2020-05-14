#include <iostream> 
using namespace std;

#define maxTOPS 14.0

double calcComputeTime(long int nM, long int nN, long int nK) {
  double numOps = nM * nN * nK;
  return numOps/(maxTOPS*1000000000.0);
}

double calcMemTime(long int nM, long int nN, long int nK) {
  double memRead = (nM * nN + nN * nK + nM * nK)/268435456.0;
  return memRead;
  // return (memRead/maxTOPS) * 1000000;
}

void printTime(long int nM, long int nN, long int nK) {
  double computeTime = calcComputeTime(nM, nN, nK);
  double memTime = calcMemTime(nM, nN, nK);
  cout << nM << "\t" << nN << nK << "\t" << "\t" << computeTime << "\t" << memTime << endl; 
}

int main() {
  printTime(250188, 1, 4096);
}
