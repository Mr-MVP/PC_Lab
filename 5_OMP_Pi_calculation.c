#include <omp.h>
#include <stdio.h>

int main() {
  double a = 0.0;
  double b = 1.0;
  int n = 1000000; 
  double h = (b - a) / n;
  double sum = 0.0;

#pragma omp parallel for
  for (int i = 1; i < n; i++) {
    double x = a + i * h;
    double temp = 4.0 / (1.0 + x * x);

#pragma omp critical
    sum += temp;
  }

  sum += (4.0 / (1.0 + a * a) + 4.0 / (1.0 + b * b)) / 2.0;
  sum *= h;

  printf("Pi value: %.15f\n", sum);

  return 0;
}
