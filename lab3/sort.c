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
    //simple_quicksort_ascending(array);
    parallell_quicksort(array, 3);

    return 0;
}


void
calculate_pivot_3(const struct array *array, int *pivot_low, int *pivot_high)
{
    int max, min, average;
    min = INT_MAX;
    max = INT_MIN;

    int length = array->length;
    int n = (int)sqrt(array->length);
    int i;
    average = 0;
    for (i = 0; i < n; ++i)
    {
        int r = rand() % length;
        int current_value = array->data[r];
        average += current_value;

        if (current_value < min)
            min = current_value;
        if (current_value > max)
            max = current_value;
    }

    average /= n;

    *pivot_low = (min + average) / 2;
    *pivot_high = (max + average) / 2;

    return;
}

void
calculate_pivot(const struct array *array, int *pivot)
{
    int sum;

    int length = array->length;
    int n = (int)sqrt(array->length);

    int i;
    for (i = 0; i < n; ++i)
    {
        int current_value = array->data[random_int(length)];
        sum += current_value;
    }

    *pivot = sum / n;
    return;
}

int random_int(int max)
{
    int high;
    int low;

    low = rand();
    high = rand();

    high = high << 16;
    high = high | low;
    high &= 0xEFFFFFFF;
    printf("random %d %d\n", high, RAND_MAX);
    return high % max;
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
    printf("array length %d\n", length);
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

void *
thread_func(void *arg)
{
    struct private_thread_args *private_args = (struct private_thread_args *) arg;
    struct shared_thread_args *t_args = private_args->shared_args;
    int i;
    for (i = private_args->start_index; i < private_args->stop_index; i++)
    {
        int value = array_get(t_args->a, i);
        if (value <= t_args->pivot_low )
            array_insert(t_args->b, fetch_and_add(&t_args->left, 1), value);
        else if (value >= t_args->pivot_high){
            array_insert(t_args->b, fetch_and_add(&t_args->right, -1), value);
        }
        else
            array_insert(t_args->middle_array, fetch_and_add(&t_args->middle, 1), value);

    }
    printf("done thread func\n");
    return NULL;
}

void *
copy_func(void *arg)
{
    struct private_thread_args *private_args = (struct private_thread_args *) arg;
    struct shared_thread_args *t_args = private_args->shared_args;
    int i;
    for (i = private_args->start_index; i < private_args->stop_index; i++)
    {
        if (i < t_args->left)
        {
            array_insert(t_args->a, array_get(t_args->b, i), i);
        }
        else if (i < t_args->right)
        {
        	array_insert(t_args->a, array_get(t_args->middle_array, i - t_args->left), i);
        }
        else
        	array_insert(t_args->a, array_get(t_args->b, i), i);
    }
    printf("copy \n");
    return NULL;
}

struct partitions
{
    int index1;
    int index2;
};

struct partitions
par_partition(struct array *array)
{
	struct partitions partitions;
    struct shared_thread_args t_args;

    calculate_pivot_3(array, &t_args.pivot_low, &t_args.pivot_high);
    printf("pivots: %d %d\n", t_args.pivot_low, t_args.pivot_high);
    t_args.left = 0;
    t_args.middle = 0;
    t_args.right = array->length - 1;
    t_args.a = array;
    t_args.b = array_alloc(array->length);
    t_args.middle_array = array_alloc(array->length);

    pthread_t threads[NB_THREADS];
    struct private_thread_args private_args[NB_THREADS];
    
    partition_array(private_args, array->length);

    int i;
    for (i = 0; i < NB_THREADS; i++)
    {
        private_args[i].shared_args = &t_args;
        private_args[i].id = i;
        pthread_create(&threads[i], NULL, &thread_func, &private_args[i]);
    }

    for (i = 0; i < NB_THREADS; i++)
    {
    	printf("random %d\n", threads[i]);
    }

    for (i = 0; i < NB_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    //parallell copy back to a.
    for (i = 0; i < NB_THREADS; i++)
    {
        private_args[i].shared_args = &t_args;
        pthread_create(&threads[i], NULL, &copy_func, &t_args);
    }

    for (i = 0; i < NB_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    partitions.index1 = t_args.left;
    partitions.index2 = t_args.right;
    printf("left %d\n", t_args.left);
    printf("right %d\n", t_args.right);
    return partitions;
}



void
parallell_quicksort(struct array *array, int threads)
{
    if (threads > 1)
    {
        struct partitions partitions;
        partitions = par_partition(array);

        struct array array1;
        struct array array2;
        struct array array3;

        int no_partitions = 3;
        printf("Vi lever!\n");
        if (no_partitions == 2)
        {
            array1.length = partitions.index1;
            array1.capacity = partitions.index1;
            array1.data = &array->data[0];
            printf("partitions 2 start: %d length 1: %d\n", 0, partitions.index1);

            array2.length = array->length - partitions.index1;
            array2.capacity = array->length - partitions.index1;
            array2.data = &array->data[partitions.index1];
            printf("partitions 2start: %d length 2: %d\n", partitions.index1, array->length - partitions.index1);

            parallell_quicksort(&array1, threads / no_partitions);
            parallell_quicksort(&array2, threads / no_partitions);
        }
        else
        {
            array1.length = partitions.index1;
            array1.capacity = partitions.index1;
            array1.data = &array->data[0];
            printf("partitions 3 start: %d length 1: %d\n", 0, partitions.index1);

            array2.length = partitions.index2 - partitions.index1;
            array2.capacity = partitions.index2 - partitions.index1;
            array2.data = &array->data[partitions.index1];
            printf("partitions 3 start: %d length 2: %d\n", partitions.index1, partitions.index2 - partitions.index1);

            array3.length = array->length - partitions.index2;
            array3.capacity = array->length - partitions.index2;
            array3.data = &array->data[partitions.index2];
            printf("partitions 3 start: %d length 2: %d\n", partitions.index2, array->length - partitions.index2);

            parallell_quicksort(&array1, threads / no_partitions);
            parallell_quicksort(&array2, threads / no_partitions);
            parallell_quicksort(&array3, threads / no_partitions);
        }
    }
    else
    {
    	simple_quicksort_ascending(array);
    }
}