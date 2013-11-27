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
sort(struct array * array)
{
	srand(time(NULL));
	//simple_quicksort_ascending(array);
	insertion_sort(array);

	return 0;
}


void
calculate_pivot_3(struct array * array, int *pivot_low, int *pivot_high)
{
	int max, min, average;
	min = INT_MAX;
	max = INT_MIN;

	int length = array->length;
	int n = (int)sqrt(array->length);
	int i;
	for (i = 0; i < n; ++i)
	{
		int current_value = array->data[random_int(length)];
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
calculate_pivot(struct array * array, int *pivot)
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
insertion_sort(struct array * array)
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
}


struct private_thread_args{
	int start_index, stop_index;
};

struct shared_thread_args{
	int pivot_low, pivot_high;
	int left, right, middle;
	struct array* b;
	struct array* a;
	struct array* middle_array;
	struct private_thread_args private_args;
};

void
partition_array(struct private_thread_args* private_args, int length){
	int i;

	private_args[0].start_index = 0;
	private_args[0].stop_index = length/NB_THREADS;

	for(i = 1; i < NB_THREADS - 1; i++){
		private_args[i].start_index = private_args[i-1].stop_index;

		if(i < NB_THREADS - 1)
			private_args[i].stop_index = (NB_THREADS+1) * length/NB_THREADS;
		else
			private_args[NB_THREADS - 1].stop_index = length;	
	}
}

void 
thread_func(void* arg)
{
	struct shared_thread_args* t_args = (struct shared_thread_args*) arg;
	int i;
	for(i = t_args->private_args.start_index; i < t_args->private_args.stop_index; i++){
		if (array_get(t_args->a, i) <= t_args->pivot_low ) 
			t_args->b[fetch_and_add(&t_args->left, 1)] = t_args->a[i];
		else if (array_get(t_args->a, i) >= t_args->pivot_high)
			t_args->b[ fetch_and_add(&t_args->right, -1)] = t_args->a[i];
		else
			t_args->middle_array[fetch_and_add(&t_args->middle, 1)] = t_args->a[i];
	}
}

void
copy_func(void* arg)
{
	struct shared_thread_args* t_args = (struct shared_thread_args*) arg;
	int i;
	for(i = t_args->private_args.start_index; i < t_args->private_args.stop_index; i++){
		if(i < t_args->left){
			t_args->a[i] = t_args->b[i];
		}else if(i < t_args->right){
			t_args->a[i] = t_args->middle_array[i - args->left];
		}else
			t_args->a[i] = t_args->b[i];
	}		
}


void
par_partition(struct array* array, const int pivot_high, const int pivot_low, int* low_index, int* high_index)
{
	
	struct shared_thread_args t_args;
	t_args.pivot_low =  pivot_low;
	t_args.pivot_high =  pivot_high;
	t_args.left = 0;
	t_args.middle = 0;
	t_args.right = array->length - 1;
	t_args.b = array_alloc(array->length);
	t_args.middle = array_alloc(array->length);

	pthread_t threads[NB_THREADS];
	struct private_thread_args private_args[NB_THREADS];
	partition_array(private_args, array->length);
	
	int i;
	for(i = 0; i < NB_THREADS; i++)
	{
		t_args.private_args = private_args[i];
		pthread_create(threads[i], NULL, thread_func, &t_args);
	}

	for(i = 0; i < NB_THREADS; i++){
		pthread_join(threads[i]);
	}

	//parallell copy back to a.
	for(i = 0; i < NB_THREADS; i++)
	{
		t_args.private_args = private_args[i];
		pthread_create(threads[i], NULL, copy_func, &t_args);
	}

	for(i = 0; i < NB_THREADS; i++){
		pthread_join(threads[i]);
	}

	*low_index = t_args.left;
	*high_index = t_args.right;
}