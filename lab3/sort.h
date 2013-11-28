#include "array.h"

#ifndef DEBUG
#define NDEBUG
#endif

#ifndef SORT
#define SORT

int sort(struct array *);

inline int fetch_and_add(int *ptr, int value);
void calculate_pivot_3(const struct array * array, int *pivot_low, int *pivot_high);
void calculate_pivot(const struct array * array, int *pivot);
int random_int(int max);
void insertion_sort(struct array * array);
void parallell_quicksort(struct array *array, int threads);
void parallell_samplesort(struct array *array);
#endif
