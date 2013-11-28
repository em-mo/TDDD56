#include "array.h"

#ifndef DEBUG
#define NDEBUG
#endif

#ifndef SORT
#define SORT

int sort(struct array *);

inline int fetch_and_add(int *ptr, int value);

void calculate_pivot(const struct array * array, int *pivot);
void sequential_quick_sort(struct array*);
void insertion_sort(struct array * array);

void parallell_quicksort(struct array *array, int threads);
void parallell_samplesort(struct array *array);

#endif
