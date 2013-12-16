/*
 * Placeholder for wavelet transform.
 * Currently just a simple invert.
 */

 #define WORK_SIZE 256

__kernel void kernelmain(__global unsigned char *image, __global unsigned char *data, const unsigned int width)
{
	int x, y;
	x = get_global_id(0) * 2;
	y = get_global_id(1) * 2;
	unsigned int index1 = y * (width * 3) + (x * 3);
	unsigned int index2 = y * (width * 3) + ((x + 1) * 3);
	unsigned int index3 = (y + 1) * (width * 3) + (x * 3);
	unsigned int index4 = (y + 1) * (width * 3) + ((x + 1) * 3);

	// __local unsigned char buffer[WORK_SIZE * 4 * 3];

	// int local_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
	// int offsetX = get_global_id(0) - get_local_id(0);
	// int offsetY = get_global_id(1) - get_local_id(1);
	// int offset = offsetY * width * 3 + offsetX * 3;
	// int i;
	
	// for(i = 1; i <= 6; i++)
	// {
		// buffer[i * WORK_SIZE + local_id] = image[offset + i * WORK_SIZE + local_id];
		// buffer[i * WORK_SIZE + local_id + get_local_size(0) * 6] = image[offset + i * WORK_SIZE + local_id + (width * 3)];
	// }
 
	data[index1 + 0] = (image[index1 + 0] + image[index2 + 0] + image[index3 + 0] + image[index4 + 0]) / 4;
	data[index1 + 1] = (image[index1 + 1] + image[index2 + 1] + image[index3 + 1] + image[index4 + 1]) / 4;
	data[index1 + 2] = (image[index1 + 2] + image[index2 + 2] + image[index3 + 2] + image[index4 + 2]) / 4;

	data[index2 + 0] = (image[index1 + 0] + image[index2 + 0] - image[index3 + 0] - image[index4 + 0]) / 4;
	data[index2 + 1] = (image[index1 + 1] + image[index2 + 1] - image[index3 + 1] - image[index4 + 1]) / 4;
	data[index2 + 2] = (image[index1 + 2] + image[index2 + 2] - image[index3 + 2] - image[index4 + 2]) / 4;

	data[index3 + 0] = (image[index1 + 0] - image[index2 + 0] + image[index3 + 0] - image[index4 + 0]) / 4;
	data[index3 + 1] = (image[index1 + 1] - image[index2 + 1] + image[index3 + 1] - image[index4 + 1]) / 4;
	data[index3 + 2] = (image[index1 + 2] - image[index2 + 2] + image[index3 + 2] - image[index4 + 2]) / 4;

	data[index4 + 0] = (image[index1 + 0] - image[index2 + 0] - image[index3 + 0] + image[index4 + 0]) / 4;
	data[index4 + 1] = (image[index1 + 1] - image[index2 + 1] - image[index3 + 1] + image[index4 + 1]) / 4;
	data[index4 + 2] = (image[index1 + 2] - image[index2 + 2] - image[index3 + 2] + image[index4 + 2]) / 4;
}
