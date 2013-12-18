/*
 * Placeholder for wavelet transform.
 * Currently just a simple invert.
 */

#define WORK_WIDTH 16
#define WORK_SIZE (WORK_WIDTH * WORK_WIDTH)

__kernel void kernelmain(__global unsigned char *image, __global unsigned char *data, const unsigned int width)
{
	int x, y, k, l;
	l = get_global_id(0);
	k = get_global_id(1);

	x = l * 2;
	y = k * 2;
	unsigned int index1 = y * (width * 3) + (x * 3);
	unsigned int index2 = y * (width * 3) + ((x + 1) * 3);
	unsigned int index3 = (y + 1) * (width * 3) + (x * 3);
	unsigned int index4 = (y + 1) * (width * 3) + ((x + 1) * 3);
 
	unsigned int outdex1 = k * (width * 3) + (l * 3);
	unsigned int outdex2 = outdex1 + (width * 3) / 2;
	unsigned int outdex3 = outdex1 + (height * 3) / 2;
	unsigned int outdex4 = outdex1 + (width * 3) / 2 + (height * 3);

	data[outdex1 + 0] = (image[index1 + 0] + image[index2 + 0] + image[index3 + 0] + image[index4 + 0]) / 4;
	data[outdex1 + 1] = (image[index1 + 1] + image[index2 + 1] + image[index3 + 1] + image[index4 + 1]) / 4;
	data[outdex1 + 2] = (image[index1 + 2] + image[index2 + 2] + image[index3 + 2] + image[index4 + 2]) / 4;

	data[outdex2 + 0] = (image[index1 + 0] + image[index2 + 0] - image[index3 + 0] - image[index4 + 0]) / 4;
	data[outdex2 + 1] = (image[index1 + 1] + image[index2 + 1] - image[index3 + 1] - image[index4 + 1]) / 4;
	data[outdex2 + 2] = (image[index1 + 2] + image[index2 + 2] - image[index3 + 2] - image[index4 + 2]) / 4;

	data[outdex3 + 0] = (image[index1 + 0] - image[index2 + 0] + image[index3 + 0] - image[index4 + 0]) / 4;
	data[outdex3 + 1] = (image[index1 + 1] - image[index2 + 1] + image[index3 + 1] - image[index4 + 1]) / 4;
	data[outdex3 + 2] = (image[index1 + 2] - image[index2 + 2] + image[index3 + 2] - image[index4 + 2]) / 4;

	data[outdex4 + 0] = (image[index1 + 0] - image[index2 + 0] - image[index3 + 0] + image[index4 + 0]) / 4;
	data[outdex4 + 1] = (image[index1 + 1] - image[index2 + 1] - image[index3 + 1] + image[index4 + 1]) / 4;
	data[outdex4 + 2] = (image[index1 + 2] - image[index2 + 2] - image[index3 + 2] + image[index4 + 2]) / 4;
}
