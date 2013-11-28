/*
 * sort.c
 *
 *  Created on: 5 Sep 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 *
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 *
 */

// Do not touch or move these lines
#include <stdio.h>
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "disable.h"
#include "pthread.h"

#ifndef DEBUG
#define NDEBUG
#endif

#include "array.h"
#include "merge_sort.h"
#include "simple_quicksort.h"

inline int
fetch_and_add(int *ptr, int value)
{
    return __sync_fetch_and_add(ptr, value);
}



int
sort(struct array *array)
{
    simple_quicksort_ascending(array);

    return 0;
}

struct loop_bounds
{
    int lower;
    int upper;
};
struct shared_thread_args
{
    struct array *a;
    struct array *b;
    struct array *c;
};
struct private_thread_args
{
    struct loop_bounds bounds_a, bounds_b;
    int id;
    struct shared_thread_args *shared_args;
};

int
binary_helper(struct array * array, int value, int lower, int upper)
{
    current_index = (upper + lower) / 2;
    current_compare = array_get(array, current_index);

    if (upper - lower == 0)
        if (value <= current_compare)
            return current_index;
        else
            return current_index + 1;
    else if (value <= current_compare)
        return binary_helper(array, value, lower, current_index);
    else
        return binary_helper(array, value, current_index + 1, upper);
}

inline int
binary_rank_search(struct array * array, int value)
{
    return binary_helper(array, value, 0, array->length - 1);
}

void
partition_array(struct loop_bounds *bounds, int length)
{
    int i;

    bounds[0].lower = 0;
    bounds[0].upper = length / NB_THREADS;
    printf("0) start: %d ", bounds[0].lower);
    printf("0 stop: %d\n", bounds[0].upper);
    for (i = 1; i < NB_THREADS; i++)
    {
        bounds[i].lower = bounds[i - 1].upper;

        if (i < NB_THREADS)
            bounds[i].upper = (i + 1) * length / NB_THREADS;
        else
            bounds[NB_THREADS - 1].upper = length;

        printf("%d) start: %d ", i, bounds[i].lower);
        printf("%d stop: %d\n", i, bounds[i].upper);
    }
    return;
}

void*
thread_merge(void* arg)
{
    struct private_thread_args *private_args = (struct private_thread_args *)arg;
    struct shared_thread_args *shared_args = private_args->shared_args;

    struct loop_bounds bounds_a, bounds_b;
    bounds_a = private_args->bounds_a;
    bounds_b = private_args->bounds_b;

    int rank, value, i;
    for (i = bounds_a.lower; i < bounds_a.upper; ++i)
    {
        value = array_get(shared_args->a, i);
        rank = binary_rank_search(shared_args->b, value);
        array_insert(shared_args->c, value, i + rank);
    }

    for (i = bounds_b.lower; i < bounds_b.upper; ++i)
    {
        value = array_get(shared_args->b, i);
        rank = binary_rank_search(shared_args->a, value);
        array_insert(shared_args->c, value, i + rank);   
    }
    return;
}

void
parallell_merge(struct array * a, struct array * b, struct array * c)
{
    struct private_thread_args private_args[NB_THREADS];
    struct shared_thread_args shared_args;
    struct loop_bounds loop_bounds_A[NB_THREADS];
    struct loop_bounds loop_bounds_B[NB_THREADS];
    
    partition_array(&loop_bounds_A, a->length);
    partition_array(&loop_bounds_B, b->length);

    shared_args.a = a;
    shared_args.b = b;
    shared_args.c = c;

    pthread_t threads[NB_THREADS];
    int i;
    for (i = 0; i < NB_THREADS; ++i)
    {
        private_args[i].shared_args = &shared_args;
        private_args[i].bounds_a = loop_bounds_A[i];
        private_args[i].bounds_b = loop_bounds_B[i];
        private_args[i].id = i;
        pthread_create(&threads[i], NULL, &thread_merge, &private_args[i]);
    }

    for (i = 0; i < NB_THREADS; ++i)
    {
        pthread_join(threads[i], NULL);
    }
    return;
}

void
construct_arrays(struct array * array, struct array *constructing_arrays, struct loop_bounds *loop_bounds, int count)
{
    int i;
    for (i = 0; i < count; ++i)
    {
        constructing_arrays[i].length = loop_bounds[i].upper - loop_bounds[i].lower;
        constructing_arrays[i].capacity = constructing_arrays[i].length;
        constructing_arrays[i].data = &array->data[loop_bounds[i].lower];
    }
}

void*
thread_sequential_sort(void* arg)
{
    struct array *array = (struct array *)arg;
    simple_quicksort_ascending(array);
}

void parallell_merge_sort(struct array * array)
{
    struct loop_bounds loop_bounds[NB_THREADS];
    struct array thread_arrays[NB_THREADS];

    partition_array(&loop_bounds, array->length);
    construct_arrays(array, thread_arrays, loop_bounds, NB_THREADS);

    pthread_t threads[NB_THREADS];
    int i;
    for (i = 0; i < NB_THREADS; ++i)
    {
        pthread_create(&threads, NULL, &thread_sequential_sort, &thread_arrays[i]);
    }

    for (i = 0; i < NB_THREADS; ++i)
    {
        pthread_join(threads, NULL);
    }

    if (NB_THREADS > 2)
    {
        struct array *tmp_array1, *tmp_array2, *swap_array;
        tmp_array1 = array_alloc(array->length);
        tmp_array2 = array_alloc(array->length);
    
        parallell_merge(&thread_arrays[0], &thread_arrays[1], tmp_array1);
        for (i = 2; i < NB_THREADS - 1; ++i)
        {
            parallell_merge(&thread_arrays[i], tmp_array1, tmp_array2);
            swap_array = tmp_array1;
            tmp_array1 = tmp_array2;
            tmp_array2 = swap_array;
        }
        parallell_merge(&thread_arrays[NB_THREADS - 1], tmp_array1, array);
    }
    else
        parallell_merge(&thread_arrays[0], &thread_arrays[1], array);


    return;
}