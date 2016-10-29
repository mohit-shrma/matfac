#include <iostream>
#include <vector>

/* Parameters */
#define M 6
#define N 4
#define LDA M
#define LDU M
#define LDVT N


std::vector<double> vadd(std::vector<double> a, std::vector<double> b) {
  std::vector<double> c(a.size(), 0);
  for (int i = 0; i < a.size(); i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}

void vaddInPlace(std::vector<double>& a, std::vector<double>& b) {
  for (int i = 0; i < a.size(); i++) {
    a[i] += b[i];
   }
 }

#pragma omp declare reduction(vecAdd: std::vector<double>: omp_out=vadd(omp_out,omp_in)) initializer(omp_priv=std::vector<double>(10, 0))

void customRedTest() {

  std::vector<double> m(10, 0);
  std::vector<double> n(10, 1);
  int k = 0;
#pragma omp parallel for reduction(+: k), reduction(vecAdd: m)
  for (int i = 0; i < 50; i++) {
    vaddInPlace(m, n); 
    k++;
  }
  
  for (auto&& v: m) {
    std::cout << v << ",";
  }
  std::cout << "k = " << k << std::endl;
}


int main(int argc, char *argv[]) {

  customRedTest();
  return 0;
}


