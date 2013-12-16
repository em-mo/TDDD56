/*
 * Rank sorting in sorting OpenCL
 * This kernel has a bug. What?
 */
#define WORK_SIZE 256

__kernel void sort(__global unsigned int *data, __global unsigned int *out, const unsigned int length)
{ 
    unsigned int pos = 0;
    unsigned int i, j;
    unsigned int val;
  
      //find out how many values are smaller
    __local unsigned int buffer[WORK_SIZE];
  	

    val = data[get_global_id(0)];

    for (j = 0; j < length; j += WORK_SIZE)
    {	
    	buffer[get_local_id(0)] = data[j+get_local_id(0)];
    	barrier(CLK_GLOBAL_MEM_FENCE);
		for (i = 0; i < WORK_SIZE; i++)
        	if (val > buffer[i])
      			pos++;
      	barrier(CLK_GLOBAL_MEM_FENCE);
  	}

    out[pos]=val;
}
