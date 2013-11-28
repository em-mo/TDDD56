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
#include "sort.h"
#include "simple_quicksort.h"

#define INT_MAX 2147483647
#define INT_MIN -2147483647

inline int
fetch_and_add(int *ptr, int value)
{
    return __sync_fetch_and_add(ptr, value);
}

int
sort(struct array *array)
{
    srand(time(NULL));
    simple_quicksort_ascending(array);

    return 0;
}


void
calculate_pivot_for_threads(const struct array *array, int *pivot)
{
    struct array * tmp_array;

    int length = array->length;
    int n = (int)sqrt(array->length);
    int i;

    tmp_array = array_alloc(n);

    for (i = 0; i < n; ++i)
    {
        int r = rand() % length;
        int current_value = array->data[r];
        
        array_put(tmp_array, current_value);

    }

    simple_quicksort_ascending(tmp_array);

    int step = n / NB_THREADS;
    for (i = 1; i < NB_THREADS; ++i)
    {
        pivot[i - 1] = array_get(tmp_array, i * step);
    }

    array_free(tmp_array);
    return;
}

inline void
swap(int *a, int *b)
{
    int c;
    c = *a;
    *a = *b;
    *b = c;
    return;
}

int
quicksort_pivot(struct array *array)
{
    int samples[3];

    samples[0] = array_get(array, 0);
    samples[1] = array_get(array, array->length / 2);
    samples[2] = array_get(array, array->length - 1);

    //sort
    if (samples[0] > samples[1]) 
        swap(&samples[0], &samples[1]);
    if (samples[1] > samples[2]) 
    {
        swap(&samples[1], &samples[2]);
        if (samples[0] > samples[1]) 
            swap(&samples[0], &samples[1]);
    }


    return samples[1];
}



void
insertion_sort(struct array *array)
{
    int i;
    for (i = 1; i < array->length; ++i)
    {
        int c;
        for (c = i; c > 0 && array->data[c] > array->data[c - 1]; --c)
        {
            swap(&array->data[c], &array->data[c - 1]);
        }

    }
    return;
}

int
sequential_partition(struct array *array, int pivot)
{
    int current_index = 0, i;
    for (i = 0; i < array->length; ++i)
    {
        if (array_get(array, i) <= pivot)
        {
            swap(&array->data[i], &array->data[current_index++]);
        }
    }
    return current_index;
}

void
sequential_quick_sort(struct array *array)
{
    if (array->length < 10)
        insertion_sort(array);
    else
    {
        int split_index = sequential_partition(array, quicksort_pivot(array));
        struct array left_array, right_array;

        left_array.length = split_index;
        left_array.capacity = split_index;
        left_array.data = array->data;

        right_array.length = array->length - split_index;
        right_array.capacity = array->length - split_index;
        right_array.data = &array->data[split_index];

        sequential_quick_sort(&left_array);
        sequential_quick_sort(&right_array);
    }
}

struct shared_thread_args
{
    int pivot_low, pivot_high;
    int left, right, middle;
    struct array *b;
    struct array *a;
    struct array *middle_array;
};
struct private_thread_args
{
    int start_index, stop_index;
    int id;
    struct shared_thread_args *shared_args;
};

void
partition_array(struct private_thread_args *private_args, int length)
{
    int i;

    private_args[0].start_index = 0;
    private_args[0].stop_index = length / NB_THREADS;
 	printf("0) start: %d ", private_args[0].start_index);
    printf("0 stop: %d\n", private_args[0].stop_index);
    for (i = 1; i < NB_THREADS; i++)
    {
        private_args[i].start_index = private_args[i - 1].stop_index;

        if (i < NB_THREADS)
            private_args[i].stop_index = (i + 1) * length / NB_THREADS;
        else
            private_args[NB_THREADS - 1].stop_index = length;

        printf("%d) start: %d ", i, private_args[i].start_index);
        printf("%d stop: %d\n", i, private_args[i].stop_index);
    }
    return;
}

