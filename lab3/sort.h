#include "array.h"

#ifndef DEBUG
#define NDEBUG
#endif

#ifndef SORT
#define SORT

int sort(struct array *);

inline int fetch_and_add(int *ptr, int value);
void calculate_pivot_3(struct array * array, int *pivot_low, int *pivot_high);
void calculate_pivot(struct array * array, int *pivot);
int random(int max);
void insertion_sort(struct array * array);
#endif
