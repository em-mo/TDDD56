// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <math.h>

const int N = 16; 
const int blocksize = 16; 

__global__ 
void simple(float *c) 
{
	c[threadIdx.x] = sqrt(c[threadIdx.x]);
}

int main()
{
	float *original = new float[N];
	float *target = new float[N];
	float *cd;
	const int size = N*sizeof(float);
	
	cudaMalloc( (void**)&cd, size );
	cudaMemcpy( cd, original, size, cudaMemcpyHostToDevice ); 
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	simple<<<dimGrid, dimBlock>>>(cd);
	cudaThreadSynchronize();
	cudaMemcpy( target, cd, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cd );
	
	for (int i = 0; i < N; i++)
	{
		printf("%f ", original[i]);
	}
	printf("\n");
	for (int i = 0; i < N; i++)
	{
		printf("%f ", target[i]);
	}
	printf("\n");
	delete[] original;
	delete[] target;
	printf("done\n");
	return EXIT_SUCCESS;
}
