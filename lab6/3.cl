/*
 * Placeholder for wavelet transform.
 * Currently just a simple invert.
 */

#define WORK_WIDTH 16
#define WORK_SIZE (WORK_WIDTH * WORK_WIDTH)

__kernel void kernelmain(__global unsigned char *image, __global unsigned char *data, const unsigned int width)
{
	int x, y;
	x = get_global_id(0) * 2;
	y = get_global_id(1) * 2;
	unsigned int index1 = y * (width * 3) + (x * 3);
	unsigned int index2 = y * (width * 3) + ((x + 1) * 3);
	unsigned int index3 = (y + 1) * (width * 3) + (x * 3);
	unsigned int index4 = (y + 1) * (width * 3) + ((x + 1) * 3);

	__local unsigned char buffer[WORK_SIZE * 4 * 3];

	int local_id = get_local_id(1) * WORK_WIDTH + get_local_id(0);
	int offsetX = get_global_id(0) - get_local_id(0);
	int offsetY = get_global_id(1) - get_local_id(1);
	int offset = offsetY * width * 3 + offsetX * 3;
	int i;
	
	for(i = 0; i < 6; i++)
	{
		buffer[i * WORK_SIZE + local_id] = image[offset + i * WORK_SIZE + local_id];
		buffer[i * WORK_SIZE + local_id + WORK_WIDTH * 6] = 
							image[offset + i * WORK_SIZE + local_id + (width * 3)];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	x = get_local_id(0) * 2;
	y = get_local_id(1) * 2;
	unsigned int buf_id1 = y * (WORK_WIDTH * 6) + (x * 3);
	unsigned int buf_id2 = y * (WORK_WIDTH * 6) + ((x + 1) * 3);
	unsigned int buf_id3 = (y + 1) * (WORK_WIDTH * 6) + (x * 3);
	unsigned int buf_id4 = (y + 1) * (WORK_WIDTH * 6) + ((x + 1) * 3);
 
	data[index1 + 0] = (buffer[buf_id1 + 0] + buffer[buf_id2 + 0] + buffer[buf_id3 + 0] + buffer[buf_id4 + 0]) / 4;
	data[index1 + 1] = (buffer[buf_id1 + 1] + buffer[buf_id2 + 1] + buffer[buf_id3 + 1] + buffer[buf_id4 + 1]) / 4;
	data[index1 + 2] = (buffer[buf_id1 + 2] + buffer[buf_id2 + 2] + buffer[buf_id3 + 2] + buffer[buf_id4 + 2]) / 4;

	data[index2 + 0] = (buffer[buf_id1 + 0] + buffer[buf_id2 + 0] - buffer[buf_id3 + 0] - buffer[buf_id4 + 0]) / 4;
	data[index2 + 1] = (buffer[buf_id1 + 1] + buffer[buf_id2 + 1] - buffer[buf_id3 + 1] - buffer[buf_id4 + 1]) / 4;
	data[index2 + 2] = (buffer[buf_id1 + 2] + buffer[buf_id2 + 2] - buffer[buf_id3 + 2] - buffer[buf_id4 + 2]) / 4;

	data[index3 + 0] = (buffer[buf_id1 + 0] - buffer[buf_id2 + 0] + buffer[buf_id3 + 0] - buffer[buf_id4 + 0]) / 4;
	data[index3 + 1] = (buffer[buf_id1 + 1] - buffer[buf_id2 + 1] + buffer[buf_id3 + 1] - buffer[buf_id4 + 1]) / 4;
	data[index3 + 2] = (buffer[buf_id1 + 2] - buffer[buf_id2 + 2] + buffer[buf_id3 + 2] - buffer[buf_id4 + 2]) / 4;

	data[index4 + 0] = (buffer[buf_id1 + 0] - buffer[buf_id2 + 0] - buffer[buf_id3 + 0] + buffer[buf_id4 + 0]) / 4;
	data[index4 + 1] = (buffer[buf_id1 + 1] - buffer[buf_id2 + 1] - buffer[buf_id3 + 1] + buffer[buf_id4 + 1]) / 4;
	data[index4 + 2] = (buffer[buf_id1 + 2] - buffer[buf_id2 + 2] - buffer[buf_id3 + 2] + buffer[buf_id4 + 2]) / 4;
}
