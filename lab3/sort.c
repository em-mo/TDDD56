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
	simple_quicksort_ascending(array);

	return 0;
}

void
calculate_pivot_3(struct * array, int *pivot_low, int *pivot_high)
{
	int max, min, average;
	min = INT_MAX;
	max = INT_MIN;

	int length = array->length;
	int n = (int)sqrt(array->length);

	for (int i = 0; i < n; ++i)
	{
		int current_value = array->data[random(length)]
		average += current_value;

		if (current_value < min)
			min = current_value;
		if (current_value > max)
			max = current_value;
	}

	average /= n;

	pivot_low = (min + average) / 2;
	pivot_high = (max + average) / 2;
}

void
calculate_pivot(struct * array, int *pivot)
{
	int sum;

	int length = array->length;
	int n = (int)sqrt(array->length);

	for (int i = 0; i < n; ++i)
	{
		int current_value = array->data[random(length)]
		average += current_value;
	}

	pivot = sum / n;
}

int
random(int max)
{
	int high;
	int low;

	low = rand();
	high = rand();

	high = high << 16;
	high = high | low;

	return high % max;
}