// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>
#include "milli.h"
__global__ 
void simple(float *a, float *b, float *c, int N) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int index = idx + idy*N;

	if(idx < N && idy < N)
		c[index] = a[index] + b[index];
}

__global__ 
void simpleFlipped(float *a, float *b, float *c, int N) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int index = idy + idx*N;

	if(idx < N && idy < N)
		c[index] = a[index] + b[index];
}

void add_matrix(float *a, float *b, float *c, int N)
{
	int index;
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = j + i*N;
			c[index] = a[index] + b[index];
		}
}

int main(int argc, char** argv)
{ 
	int N = 512;
	int gridX = 32;
	int gridY = 32;

	if(argc > 0)
		N = atoi(argv[1]);

	if(argc > 1)
	{
		int gridSize = atoi(argv[2]);
		gridX = gridSize;
		gridY = gridSize;
	}
	float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];
	float *c2 = new float[N*N];

	float* ad;
	float* bd;
	float* cd;

	int size = N*N* sizeof(float);

	cudaEvent_t start_event;
	cudaEvent_t end_event;
	float theTime, theTime2;

	cudaMalloc( (void**)&ad, size );
	cudaMalloc( (void**)&bd, size );
	cudaMalloc( (void**)&cd, size );

	cudaEventCreate(&start_event);
	cudaEventCreate(&end_event);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}
	}

	cudaMemcpy( ad, a, size, cudaMemcpyHostToDevice ); 
	cudaMemcpy( bd, b, size, cudaMemcpyHostToDevice ); 

	dim3 dimBlock((N+1)/gridX, (N+1)/gridY);
	dim3 dimGrid( gridX, gridY );

	cudaEventRecord(start_event, 0);
	cudaEventSynchronize(start_event);
	
	simple<<<dimGrid, dimBlock>>>(ad, bd, cd, N);

	cudaThreadSynchronize();
  	cudaEventRecord(end_event, 0);
	cudaEventSynchronize(end_event);
  	
	cudaEventElapsedTime(&theTime, start_event, end_event);

	cudaEventRecord(start_event, 0);
	cudaEventSynchronize(start_event);
	
	simpleFlipped<<<dimGrid, dimBlock>>>(ad, bd, cd, N);

	cudaThreadSynchronize();
  	cudaEventRecord(end_event, 0);
	cudaEventSynchronize(end_event);
	cudaEventElapsedTime(&theTime2, start_event, end_event);

	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 	

	/*for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}*/	
	int t1 = GetMicroseconds();
	add_matrix(a, b, c2, N);
	int t2 = GetMicroseconds();
	for(int i = 0; i < N*N; i++)
	{
		if(c[i] != c2[i]){
			printf("olika! %f!=%f\n", c[i], c2[i]);
			break;
		}
	}

	printf("time cuda: %f us\n", theTime*1000);
	printf("time non-coalescing cuda: %f us\n", theTime2*1000);
	printf("time cpu: %d us\n", t2 - t1);
	

	//struct cudaDeviceProp prop;
	//cudaGetDeviceProperties(&prop, 0);

	//printf("device max threads: %d\n", prop.maxThreadsPerBlock);
	//printf("device warpSize: %d\n", prop.warpSize);
}
