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
     // sequential_quick_sort(array);
    // insertion_sort(array);
    //parallell_quicksort(array, 3);
    //simple_quicksort_ascending(array);
    parallell_samplesort(array);

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
        printf("skapa pivot %d\n", array_get(tmp_array, i * step));
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
    // int samples[3];

    // samples[0] = array_get(array, 0);
    // samples[1] = array_get(array, array->length / 2);
    // samples[2] = array_get(array, array->length - 1);

    // //sort
    // if (samples[0] > samples[1]) 
    //     swap(&samples[0], &samples[1]);
    // if (samples[1] > samples[2]) 
    // {
    //     swap(&samples[1], &samples[2]);
    //     if (samples[0] > samples[1]) 
    //         swap(&samples[0], &samples[1]);
    // }

    // return samples[1];

    struct array * tmp_array;

    int length = array->length;
    int n = (int)sqrt(array->length);
    int i, pivot;

    tmp_array = array_alloc(n);

    for (i = 0; i < n; ++i)
    {
        int r = rand() % length;
        int current_value = array->data[r];
        
        array_put(tmp_array, current_value);
    }

    simple_quicksort_ascending(tmp_array);
    pivot = array_get(tmp_array, tmp_array->length / 2);
    array_free(tmp_array);
    return pivot;
}



void
insertion_sort(struct array *array)
{
    int i;
    for (i = 1; i < array->length; ++i)
    {
        int c;
        for (c = i; c > 0 && array->data[c] < array->data[c - 1]; --c)
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
        int element = array_get(array, i);
        if (element == pivot)
        {
            if (rand() % 2)
                swap(&array->data[i], &array->data[current_index++]);
        }
        else if (element < pivot)
        {
            swap(&array->data[i], &array->data[current_index++]);
        }
    }
    return current_index;
}

// void
// sequential_quick_sort(struct array *array)
// {
//     if (array->length < 10)
//         insertion_sort(array);
//     else
//     {
//         int pivot = quicksort_pivot(array);

//         int split_index = sequential_partition(array, pivot);

//         struct array left_array, right_array;
        

//         left_array.length = split_index;
//         left_array.capacity = split_index;
//         left_array.data = array->data;

//         right_array.length = array->length - split_index;
//         right_array.capacity = array->length - split_index;
//         right_array.data = &array->data[split_index];

//         sequential_quick_sort(&left_array);
//         sequential_quick_sort(&right_array);
//     }
// }

void
sequential_quick_sort(struct array * array)
{
    int length = array->length;
    int *data = array->data;
    if (length < 10)
    {
        insertion_sort(array);
        return;
    }
    int pivot = data[length / 2];
    int left = 0;
    int right = length - 1;
    while (left <= right) {
        if (data[left] < pivot) {
            ++left;
            continue;
        }
        if (data[right] > pivot) {
            --right;
            continue;
        }
        int tmp = data[left];
        data[left++] = data[right];
        data[right--] = tmp;
    }
    struct array left_array, right_array;
    left_array.length = right + 1;
    left_array.capacity = right + 1;
    left_array.data = data;

    right_array.length = array->length - left;
    right_array.capacity = right_array.length;
    right_array.data = &data[left];
    sequential_quick_sort(&left_array);
    sequential_quick_sort(&right_array);
    return;
}

struct shared_thread_args
{
    int *pivot;
    int *length;
    struct array *a;
    struct array ***partitions;
    pthread_barrier_t *barrier;
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

int
calculate_start(int *length, int id)
{
    int i;
    int sum = 0;
    for(i = 0; i < id; i++)
    {
        sum += length[i];
    }
    return sum;
}   

void*
thread_func(void* arg)
{
    struct private_thread_args* private_args = (struct private_thread_args*) arg;
    struct shared_thread_args* shared_args = private_args->shared_args;

    int id = private_args->id;
    printf("id %d\n", id);
    int start_index = private_args->start_index;
    int stop_index = private_args->stop_index;

    struct array* a = shared_args->a;
    int *pivot = shared_args->pivot;

    struct array ***partitions = shared_args->partitions;
    int *length = shared_args->length;
    // Give the lone thread something to work on
    if (NB_THREADS > 1)
    {
        int i, j, value;
        for(i = start_index; i < stop_index; i++)
        {
            value = array_get(a, i);

            for(j = 0; j < NB_THREADS - 1; j++)
            {   
                if(value < pivot[j])
                {
                    array_put(partitions[id][j], value);
                    break;
                }else if(j == NB_THREADS - 2){
                    array_put(partitions[id][j+1], value);
                }
            }
        }
    }
    else
    {
        int i;
        for(i = start_index; i < stop_index; i++)
        {
            array_put(partitions[id][0], array_get(a, i));
        }
    }
    int i;
    for(i = 0; i < NB_THREADS; i++)
        fetch_and_add(&length[i], partitions[id][i]->length);

    pthread_barrier_wait(shared_args->barrier);

    int insert_start;
    int insert_index = insert_start = calculate_start(length, id);

    int pos;
    for(i = 0; i < NB_THREADS; i++)
    {
        for(pos = 0; pos < partitions[i][id]->length; pos++)
        {
            array_insert(a, array_get(partitions[i][id], pos), insert_index);
            insert_index++;
        }
    }


    if (id == 0 && NB_THREADS > 1)
    {
        printf("pivot %d\n", pivot[0]);
    }
    struct array t_array;
    t_array.data = &a->data[insert_start];
    t_array.length = length[id];
    t_array.capacity = length[id];
    sequential_quick_sort(&t_array);
    // simple_quicksort_ascending(&t_array);
    return NULL;
}

void
parallell_samplesort(struct array* array)
{
    int pivot[NB_THREADS-1];
    calculate_pivot_for_threads(array, pivot);

    struct shared_thread_args shared_args;
    struct private_thread_args private_args[NB_THREADS];

    partition_array(private_args, array->length);

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, NB_THREADS);

    int length[NB_THREADS];
    //struct array partitions[NB_THREADS][NB_THREADS];

    struct array ***partitions = (struct array***) malloc(NB_THREADS*sizeof(struct array**));
    int i,j;
    for(i = 0; i <  NB_THREADS; i++)
    {
         partitions[i] = (struct array**)malloc(NB_THREADS*sizeof(struct array*));
    }
    for(i = 0; i <  NB_THREADS; i++)
    {
        for(j = 0; j < NB_THREADS; j++)
            partitions[i][j] = array_alloc(array->length/NB_THREADS);
    }

    shared_args.pivot = pivot;
    shared_args.length = length;
    shared_args.partitions = partitions;
    shared_args.barrier = &barrier;
    shared_args.a = array;


    pthread_t threads[NB_THREADS];
    
    for(i = 0; i < NB_THREADS; i++){
        shared_args.length[i] = 0;
        private_args[i].shared_args = &shared_args;
        private_args[i].id = i;
        pthread_create(&threads[i], NULL, &thread_func, &private_args[i]);
    }


    for(i = 0; i < NB_THREADS; i++){
        pthread_join(threads[i], NULL);
    }

    for(i = 0; i <  NB_THREADS; i++)
    {
        for(j = 0; j < NB_THREADS; j++)
             array_free(partitions[i][j]);
    }
    for(i = 0; i <  NB_THREADS; i++)
    {
        free (partitions[i]);
    }
    free(partitions);

}


