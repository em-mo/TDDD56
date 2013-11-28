#include "array.h"

#ifndef DEBUG
#define NDEBUG
#endif

#ifndef MERGE_SORT
#define MERGE_SORT

int sort(struct array *);

void parallell_merge_sort(struct array * array);

#endif
