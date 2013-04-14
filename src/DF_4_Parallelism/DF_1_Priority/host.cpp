#include <stdio.h>
#include <stdlib.h>
#include <CL\cl.h>
#include "host.h"


int main()
{
	#define MEM_SIZE 40 //we will create a 2-dimensional NDrange
	#define KERNEL_NAME "myKernel.cl"
	#define MAX_SOURCE_SIZE 10000    //maximum length of the kernel source file
	#define SUB_DEVICES_NUMBER 4

	//OpenCL elements (device, platform, memory-objects....)
	cl_device_id device_id;
	cl_device_id sub_device_ids[SUB_DEVICES_NUMBER];
	cl_device_id node_A,node_B,node_C,node_D;
	cl_context context = NULL;
	cl_command_queue command_queue[SUB_DEVICES_NUMBER];
	cl_mem inputData_A = NULL;
	cl_mem outputData_A = NULL;
	cl_mem inputData_D_fromB = NULL;
	cl_mem inputData_D_fromC = NULL;
	cl_mem outputData = NULL;
	cl_program program = NULL;
	cl_kernel kernel[SUB_DEVICES_NUMBER];
	cl_platform_id platform_id = NULL;
	
	cl_int ret;     //used to store return messages from OpenCL function calls
	
	//Obtain platform and device (CPU)
	ret = clGetPlatformIDs(1, &platform_id, NULL);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);

	//----> DEVICE FISSION
	//query the device for the max number of subdevices & computeUnits
	cl_uint maxSubdevices;
	cl_uint maxComputeUnits;
	
	ret = clGetDeviceInfo(device_id,CL_DEVICE_PARTITION_MAX_SUB_DEVICES,sizeof(cl_uint),&maxSubdevices,NULL);
	ret = clGetDeviceInfo(device_id,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&maxComputeUnits,NULL);
	
	//if at least 4 subdevices can be created...
	if(maxSubdevices >= SUB_DEVICES_NUMBER)
	{
		cl_device_partition_property params[7];
		params[0] = CL_DEVICE_PARTITION_BY_COUNTS;
		params[1] = 1;
		params[2] = 1;
		params[3] = 1;
		params[4] = 1;
		params[5] = CL_DEVICE_PARTITION_BY_COUNTS_LIST_END;
		params[6] = 0; //end of list
				
		// Create the sub-devices:
		cl_uint dump;
		ret = clCreateSubDevices(device_id, params, 4, sub_device_ids, &dump);
		node_A = sub_device_ids[0];
		node_B = sub_device_ids[1];
		node_C = sub_device_ids[2];
		node_D = sub_device_ids[3];
		printf("\n(%d)Subdevices created correctly!",dump);
	}
	else
	{
		printf("\nError! Device cannot be partitioned");
		system("pause");
		exit(-1);
	}
	
	//Create context for the device
	context = clCreateContext(NULL, 4, sub_device_ids, NULL, NULL, &ret);

	//Create Command Queue and Memory Object (monodimensional buffer)
	for(int k=0;k<SUB_DEVICES_NUMBER;k++)
		command_queue[k] = clCreateCommandQueue(context, sub_device_ids[k], 0, &ret);
	
	
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
		ret = clBuildProgram(program, SUB_DEVICES_NUMBER, sub_device_ids, NULL, NULL, NULL);
		kernel[0] = clCreateKernel(program, "Kernel_A", &ret);				  
		kernel[1] = clCreateKernel(program, "Kernel_B", &ret);				  
		kernel[2] = clCreateKernel(program, "Kernel_C", &ret);				 
		kernel[3] = clCreateKernel(program, "Kernel_D", &ret);				  
		
		outputData_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR ,MEM_SIZE * sizeof(int), NULL, &ret);
		inputData_D_fromB = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR ,MEM_SIZE * sizeof(int), NULL, &ret);
		inputData_D_fromC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR ,MEM_SIZE * sizeof(int), NULL, &ret);
		outputData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR ,MEM_SIZE * sizeof(int), NULL, &ret);
		//Initializing Input Data
		size_t globalWorkSize[1] = { MEM_SIZE };
		size_t localWorkSize[1] = { 1 };
		cl_event finished_A,finished_B,finished_C,finished_D;
		int data[MEM_SIZE];
		inputData_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR ,MEM_SIZE * sizeof(int), data, &ret);
		ret = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&inputData_A); //parameters passing to the kernel function
		ret = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&outputData_A); //parameters passing to the kernel function
		
		//executing NODE A
		ret = clEnqueueNDRangeKernel(command_queue[0], kernel[0], 1, NULL, globalWorkSize, localWorkSize,0, NULL, &finished_A); 
		clWaitForEvents(1, &finished_A);

		//executing NODE B and C simultaneously
		ret = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&outputData_A); //parameters passing to the kernel function
		ret = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *)&inputData_D_fromB); //parameters passing to the kernel function
		ret = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void *)&outputData_A); //parameters passing to the kernel function
		ret = clSetKernelArg(kernel[2], 1, sizeof(cl_mem), (void *)&inputData_D_fromC); //parameters passing to the kernel function
		ret = clEnqueueNDRangeKernel(command_queue[1], kernel[1], 1, NULL, globalWorkSize, localWorkSize,0, NULL, &finished_B);
		ret = clEnqueueNDRangeKernel(command_queue[2], kernel[2], 1, NULL, globalWorkSize, localWorkSize,0, NULL, &finished_C); 
		clWaitForEvents(1, &finished_B);
		clWaitForEvents(1, &finished_C);

		//executing NODE D


		ret = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), (void *)&inputData_D_fromB); //parameters passing to the kernel function
		ret = clSetKernelArg(kernel[3], 1, sizeof(cl_mem), (void *)&inputData_D_fromC); //parameters passing to the kernel function
		ret = clSetKernelArg(kernel[3], 2, sizeof(cl_mem), (void *)&outputData); //parameters passing to the kernel function

		ret = clEnqueueNDRangeKernel(command_queue[3], kernel[3], 1, NULL, globalWorkSize, localWorkSize,0, NULL, NULL);
		
	}
	else
	{
		//Failed to open the kernel file
		printf("\nFailed to load kernel.");
	}

	//Cleaning 
	for(int k=0;k<SUB_DEVICES_NUMBER;k++)
	{
		ret = clFlush(command_queue[k]);
		ret = clFinish(command_queue[k]);
		ret = clReleaseKernel(kernel[k]);
		ret = clReleaseCommandQueue(command_queue[k]);
	}
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(inputData_A);
	ret = clReleaseMemObject(outputData_A);
	ret = clReleaseMemObject(inputData_D_fromB);
	ret = clReleaseMemObject(inputData_D_fromC);
	ret = clReleaseMemObject(outputData);

	
	ret = clReleaseContext(context);
	
	scanf("%c",NULL);
}

