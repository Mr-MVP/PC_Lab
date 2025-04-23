#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  int array_size = 10;
  int arr[array_size];
  int max_value = -1;
  srand(time(NULL));

  // Initialize the array with random values
  for (int i = 0; i < array_size; i++) {
    arr[i] = rand() % 100;
  }

  // Print the array
  printf("Array: ");
  for (int i = 0; i < array_size; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");

  // Parallel computation of max value
  #pragma omp parallel
  {
    int l_max = -1;  // Local max for each thread

    #pragma omp for
    for (int i = 0; i < array_size; i++) {
      if (arr[i] > l_max) {
        l_max = arr[i];
      }
    }

    // Critical section to update the global max_value
    #pragma omp critical
    {
      if (l_max > max_value) {
        max_value = l_max;
      }
    }
  }

  printf("Maximum value is: %d\n", max_value);
  return 0;
}

