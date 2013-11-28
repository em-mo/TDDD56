#include "array.h"

#ifndef DEBUG
#define NDEBUG
#endif

#ifndef MERGE_SORT
#define MERGE_SORT

int sort(struct array *);

inline int fetch_and_add(int *ptr, int value);


#endif
