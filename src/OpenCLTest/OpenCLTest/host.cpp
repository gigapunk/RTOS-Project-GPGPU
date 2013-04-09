#include <stdio.h>
#include <stdlib.h>
#include <CL\cl.h>
#include "host.h"


int main()
{
	#define MEM_SIZE 128 //we will use a 128 byte buffer
	#define KERNEL_NAME "myKernel.cl"
	#define MAX_SOURCE_SIZE 10000    //maximum length of the kernel source file

	//OpenCL elements (device, platform, memory-objects....)
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem inputData = NULL , resultData = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	
	cl_int ret;     //used to store return messages from OpenCL function calls
	
	//Obtain platform and device (CPU)
	ret = clGetPlatformIDs(1, &platform_id, NULL);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);

	//Create context for the device
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	//Create Command Queue and Memory Object (monodimensional buffer)
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	int numbers[MEM_SIZE];
	for(int k=0;k<MEM_SIZE;k++)
		numbers[k] = k;
	inputData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,MEM_SIZE * sizeof(int), numbers, &ret);
	resultData = clCreateBuffer(context, CL_MEM_READ_WRITE,MEM_SIZE * sizeof(int), NULL, &ret);
	
	/*Create the Kernel program and assign it to the context.
	  We will use the clCreateProgramWithSource to build a kernel from an external .cl source file*/

	//open the .cl file
	FILE *fp;
	fp = fopen(KERNEL_NAME, "r");
	if (fp != NULL)
	{
		char * sourceCode = (char*)malloc(MAX_SOURCE_SIZE);
		int source_size = fread(sourceCode, 1, MAX_SOURCE_SIZE, fp);
		program = clCreateProgramWithSource(context, 1, (const char **)&sourceCode,(const size_t *)&source_size, &ret);
		fclose(fp);

		//Now we can build and execute the Kernel
		ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
		kernel = clCreateKernel(program, "square", &ret);				  //create the kernel called "square"
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputData); //parameters passing to the kernel function
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&resultData); //parameters passing to the kernel function
		size_t globalWorkSize[1] = { MEM_SIZE };
		size_t localWorkSize[1] = { 1 };
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, globalWorkSize, localWorkSize,0, NULL, NULL); //kernel execution
		
		// Read the output buffer back to the Host
		int result [MEM_SIZE];
		ret = clEnqueueReadBuffer(command_queue, resultData, CL_TRUE, 0, MEM_SIZE * sizeof(int), result,0, NULL, NULL);

		printf("\n>>>RESULTS:\n");
		for(int k=0;k<MEM_SIZE;k++)
			printf("Position %d - result: %d\n",k,result[k]);
		
	}
	else
	{
		//Failed to open the kernel file
		printf("\nFailed to load kernel.");
	}
	
	scanf("%c",NULL);
}

