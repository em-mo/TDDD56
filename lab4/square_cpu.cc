// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <math.h>

int main()
{
	for (int i = 0; i < 16; i++)
	{
		printf("%f ", sqrt(i));
	}
	printf("\n");
	return 0;
}
