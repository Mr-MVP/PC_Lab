#include <omp.h>
#include <stdio.h>

int is_prime(int num) {
  if (num < 2) return 0;
  if (num == 2) return 1;
  if (num % 2 == 0) return 0;
  for (int i = 3; i * i <= num; i += 2) {
    if (num % i == 0)
      return 0;
  }
  return 1;
}

int main() {
  int n;
  printf("Enter n:");
  scanf("%d", &n);

  int primes[n]; 
  int count = 0;

#pragma omp parallel for schedule(dynamic) shared(primes, count)
  for (int i = 2; i <= n; i++) {
    if (is_prime(i)) {
#pragma omp critical
      primes[count++] = i;
    }
  }

  printf("Prime numbers in range is %d: ", n);
  for (int i = 0; i < count; i++) {
    printf("%d ", primes[i]);
  }
  printf("\n");

  return 0;
}