
#include <stdio.h>
#include "readppm.c"
#ifdef __APPLE__
	#include <GLUT/glut.h>
	#include <OpenGL/gl.h>
#else
	#include <GL/glut.h>
#endif

#define BLOCKSIZE 16
#define SHAREDMEMSIZE (BLOCKSIZE + 4)

__device__
inline
void 
set_pixel(unsigned char *shared_image, unsigned char *image, int shared_index, int image_index)
{ 
	shared_image[shared_index+0] = image[image_index+0];
	shared_image[shared_index+1] = image[image_index+1];
	shared_image[shared_index+2] = image[image_index+2];
}

__global__ void filter(unsigned char *image, unsigned char *out, int n, int m)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int sumx, sumy, sumz, k, l;

// printf is OK under --device-emulation
//	printf("%d %d %d %d\n", i, j, n, m);

	if (j < n && i < m)
	{
		out[(i*n+j)*3+0] = image[(i*n+j)*3+0];
		out[(i*n+j)*3+1] = image[(i*n+j)*3+1];
		out[(i*n+j)*3+2] = image[(i*n+j)*3+2];
	}

	
	if (i > 1 && i < m-2 && j > 1 && j < n-2)
		{
			// Filter kernel
			sumx=0;sumy=0;sumz=0;
			for(k=-2;k<3;k++)
				for(l=-2;l<3;l++)
				{
					sumx += image[((i+k)*n+(j+l))*3+0];
					sumy += image[((i+k)*n+(j+l))*3+1];
					sumz += image[((i+k)*n+(j+l))*3+2];
				}
			out[(i*n+j)*3+0] = sumx/25;
			out[(i*n+j)*3+1] = sumy/25;
			out[(i*n+j)*3+2] = sumz/25;
		}
}

__global__ void filter_shared(unsigned char *image, unsigned char *out, int n, int m)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int sumx, sumy, sumz, k, l;
// printf is OK under --device-emulation

//	printf("%d %d %d %d\n", i, j, n, m);
	int local_x = threadIdx.x + 2;
	int local_y = threadIdx.y + 2;

	int global_index = (j*n+i) * 3;
	int local_index = (local_y*SHAREDMEMSIZE+local_x) * 3;

	__shared__ unsigned char shared_image[SHAREDMEMSIZE*SHAREDMEMSIZE*3];
	if (j < n && i < m)
	{
		set_pixel(shared_image, image, local_index, global_index);
		set_pixel(out, shared_image, global_index, local_index);
	}
	
	// Top
	if (threadIdx.y == 0 && blockIdx.y != 0)
	{
		set_pixel(shared_image, image, local_index-(3*SHAREDMEMSIZE), global_index-(3*n));
		set_pixel(shared_image, image, local_index-(6*SHAREDMEMSIZE), global_index-(6*n));

		// Upper left
		if (threadIdx.x == 0 && blockIdx.x != 0)
		{
			set_pixel(shared_image, image, local_index-(3*SHAREDMEMSIZE)-3, global_index-(3*n)-3);
			set_pixel(shared_image, image, local_index-(3*SHAREDMEMSIZE)-6, global_index-(3*n)-6);

			set_pixel(shared_image, image, local_index-(6*SHAREDMEMSIZE)-3, global_index-(6*n)-3);
			set_pixel(shared_image, image, local_index-(6*SHAREDMEMSIZE)-6, global_index-(6*n)-6);
		}
		// Upper right
		else if (threadIdx.x == blockDim.x - 1  && blockIdx.x != gridDim.x - 1)
		{
			set_pixel(shared_image, image, local_index-(3*SHAREDMEMSIZE)+3, global_index-(3*n)+3);
			set_pixel(shared_image, image, local_index-(3*SHAREDMEMSIZE)+6, global_index-(3*n)+6);

			set_pixel(shared_image, image, local_index-(6*SHAREDMEMSIZE)+3, global_index-(6*n)+3);
			set_pixel(shared_image, image, local_index-(6*SHAREDMEMSIZE)+6, global_index-(6*n)+6);
		}
	}
	else if (threadIdx.y == blockDim.y - 1 && blockIdx.y != gridDim.y - 1)
	{
		set_pixel(shared_image, image, local_index+(3*SHAREDMEMSIZE), global_index+(3*n));
		set_pixel(shared_image, image, local_index+(6*SHAREDMEMSIZE), global_index+(6*n));
		// Lower left
		if (threadIdx.x == 0 && blockIdx.x != 0)
		{
			set_pixel(shared_image, image, local_index+(3*SHAREDMEMSIZE)-3, global_index+(3*n)-3);
			set_pixel(shared_image, image, local_index+(3*SHAREDMEMSIZE)-6, global_index+(3*n)-6);

			set_pixel(shared_image, image, local_index+(6*SHAREDMEMSIZE)-3, global_index+(6*n)-3);
			set_pixel(shared_image, image, local_index+(6*SHAREDMEMSIZE)-6, global_index+(6*n)-6);
		}
		// Lower right
		else if (threadIdx.x == blockDim.x - 1 && blockIdx.x != gridDim.x - 1)
		{
			set_pixel(shared_image, image, local_index+(3*SHAREDMEMSIZE)+3, global_index+(3*n)+3);
			set_pixel(shared_image, image, local_index+(3*SHAREDMEMSIZE)+6, global_index+(3*n)+6);

			set_pixel(shared_image, image, local_index+(6*SHAREDMEMSIZE)+3, global_index+(6*n)+3);
			set_pixel(shared_image, image, local_index+(6*SHAREDMEMSIZE)+6, global_index+(6*n)+6);
		
		}
	}
	// Left
	if (threadIdx.x == 0 && blockIdx.x != 0)
	{
		set_pixel(shared_image, image, local_index-3, global_index-3);
		set_pixel(shared_image, image, local_index-6, global_index-6);
	}
	// Right
	else if (threadIdx.x == blockDim.x - 1  && blockIdx.x != gridDim.x - 1)
	{
		set_pixel(shared_image, image, local_index+3, global_index+3);
		set_pixel(shared_image, image, local_index+6, global_index+6);
	}
	
	__syncthreads();
	
	if (i > 1 && i < m-2 && j > 1 && j < n-2)
		{
			// Filter kernel
			sumx=0;sumy=0;sumz=0;
			for(k=-2;k<3;k++)
				for(l=-2;l<3;l++)
				{
					int index = local_index+(3*k*SHAREDMEMSIZE)+3*l;
					sumx += shared_image[index+0];
					sumy += shared_image[index+1];
					sumz += shared_image[index+2];
				}
			out[global_index+0] = sumx/25;
			out[global_index+1] = sumy/25;
			out[global_index+2] = sumz/25;
		}
}



// Compute CUDA kernel and display image
void Draw()
{
	unsigned char *image, *out;
	int n, m;
	unsigned char *dev_image, *dev_out;
	cudaEvent_t start_event;
    cudaEvent_t end_event;
    float theTime;

	image = readppm("maskros512.ppm", &n, &m);
	out = (unsigned char*) malloc(n*m*3);
	
	cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);
	cudaMalloc( (void**)&dev_image, n*m*3);
	cudaMalloc( (void**)&dev_out, n*m*3);
	cudaMemset(dev_out, 0, n*m*3);
	cudaMemcpy( dev_image, image, n*m*3, cudaMemcpyHostToDevice);
	
	dim3 dimBlock( 16, 16 );
	dim3 dimGrid( 32, 32 );

	dim3 ourBlock( BLOCKSIZE, BLOCKSIZE );
	dim3 ourGrid( n / BLOCKSIZE, m / BLOCKSIZE );

    cudaEventRecord(start_event, 0);
    cudaEventSynchronize(start_event);

	filter<<<ourGrid, ourBlock>>>(dev_image, dev_out, n, m);
	cudaThreadSynchronize();
	
    cudaEventRecord(end_event, 0);
    cudaEventSynchronize(end_event);
    
    cudaEventElapsedTime(&theTime, start_event, end_event);
    printf("Slow time: %f ms\n", theTime);
	
    cudaEventRecord(start_event, 0);
    cudaEventSynchronize(start_event);

	filter_shared<<<ourGrid, ourBlock>>>(dev_image, dev_out, n, m);
	cudaThreadSynchronize();
	
    cudaEventRecord(end_event, 0);
    cudaEventSynchronize(end_event);
    
    cudaEventElapsedTime(&theTime, start_event, end_event);
    printf("The time: %f ms\n", theTime);

	cudaMemcpy( out, dev_out, n*m*3, cudaMemcpyDeviceToHost );
	cudaFree(dev_image);
	cudaFree(dev_out);
    cudaEventDestroy(start_event);
    cudaEventDestroy(end_event);
	
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );
	glRasterPos2f(-1, -1);
	glDrawPixels( n, m, GL_RGB, GL_UNSIGNED_BYTE, image );
	glRasterPos2i(0, -1);
	glDrawPixels( n, m, GL_RGB, GL_UNSIGNED_BYTE, out );
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
	glutInitWindowSize( 1024, 512 );
	glutCreateWindow("CUDA on live GL");
	glutDisplayFunc(Draw);
	
	glutMainLoop();
}
