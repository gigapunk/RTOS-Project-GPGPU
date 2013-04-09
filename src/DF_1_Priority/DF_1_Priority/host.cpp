#include <stdio.h>
#include <stdlib.h>
#include <CL\cl.h>
#include "host.h"


int main()
{
	#define MEM_SIZE 30 //work-size for each kernel
	#define KERNEL_NAME "myKernel.cl"
	#define MAX_SOURCE_SIZE 10000    //maximum length of the kernel source file
	#define SUB_DEVICES_NUMBER 2

	//OpenCL elements (device, platform, memory-objects....)
	cl_device_id device_id;
	cl_device_id sub_device_ids[SUB_DEVICES_NUMBER];
	cl_device_id device_highPriority = NULL;
	cl_device_id device_normalPriority = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue_LO = NULL; //queue for low priority commands
	cl_command_queue command_queue_HI = NULL; //queue for high priority commands
	cl_mem inputData_true = NULL,inputData_false = NULL;
	cl_program program = NULL;
	cl_kernel kernel_A = NULL;
	cl_kernel kernel_B = NULL;
	cl_kernel kernel_C = NULL;
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
	
	//if at least 2 subdevices can be created...
	if(maxSubdevices >= 2)
	{

		cl_device_partition_property params[5];
		params[0] = CL_DEVICE_PARTITION_BY_COUNTS; 
		params[1] = 1; // 1 CU reserved for high-priority computation
		params[2] = maxComputeUnits - 1; // remaining CUs used for normal computation
		params[3] = CL_DEVICE_PARTITION_BY_COUNTS_LIST_END; // End Count list
		params[4] = 0; // End of the property list

		
		// Create the sub-devices:
		cl_uint dump;
		ret = clCreateSubDevices(device_id, params, SUB_DEVICES_NUMBER, sub_device_ids, &dump);
		device_highPriority = sub_device_ids[0];
	    device_normalPriority =  sub_device_ids[1];
		printf("\n(%d)Subdevices created correctly!",dump);
	}
	else
	{
		printf("\nError! Device cannot be partitioned");
		exit(-1);
	}
	
	//Create context for the device
	context = clCreateContext(NULL, 2, sub_device_ids, NULL, NULL, &ret);

	//Create Command Queue and Memory Object (monodimensional buffer)
	command_queue_LO = clCreateCommandQueue(context, device_normalPriority, 0, &ret);
	command_queue_HI = clCreateCommandQueue(context, device_highPriority, 0, &ret);
	int arg_false[MEM_SIZE];
	int arg_true[MEM_SIZE];
	for(int k=0;k<MEM_SIZE;k++)
	{
		arg_false[k] = 0;
		arg_true[k] = 1;
	}
	inputData_true = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR ,MEM_SIZE * sizeof(int), arg_true, &ret);
	inputData_false = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR ,MEM_SIZE * sizeof(int), arg_false, &ret);


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
		ret = clBuildProgram(program, 2, sub_device_ids, NULL, NULL, NULL);
		kernel_A = clCreateKernel(program, "lowPriorityTask", &ret);				  //create the kernel called "lowPriorityTask"
		kernel_B = clCreateKernel(program, "highPriorityTask", &ret);				  //create the kernel called "highPriorityTask"
		kernel_C = clCreateKernel(program, "highPriorityTask", &ret);				  //create the kernel called "highPriorityTask"
		//ret = clSetKernelArg(kernel_A, 0, sizeof(cl_mem), (void *)&inputData_true); //parameters passing to the kernel function
		ret = clSetKernelArg(kernel_B, 0, sizeof(cl_mem), (void *)&inputData_true); //parameters passing to the kernel function
		ret = clSetKernelArg(kernel_C, 0, sizeof(cl_mem), (void *)&inputData_false); //parameters passing to the kernel function

		size_t globalWorkSize[1] = { MEM_SIZE };
		size_t localWorkSize[1] = { 1 };
		cl_event finished_C,finished_B;
		
		/* First run: we'll execute low priority calls over the LO-device. Half of the higher priority tasks (Kernel_B) will be dispatched
		  to the HI command-queue and will be executed immediately, while others will be dispatched the LO-device and will have to wait
		  until Kernel_A has finished.*/
		printf("\n\n ---With Device Fission: ---");
		ret = clEnqueueNDRangeKernel(command_queue_LO, kernel_A, 1, NULL, globalWorkSize, localWorkSize,0, NULL, NULL); 
	    ret = clEnqueueNDRangeKernel(command_queue_HI, kernel_B, 1, NULL, globalWorkSize, localWorkSize,0, NULL, &finished_B); 
	    ret = clEnqueueNDRangeKernel(command_queue_LO, kernel_C, 1, NULL, globalWorkSize, localWorkSize,0, NULL, &finished_C); 
		
		//wait until completion
		clWaitForEvents(1, &finished_C);
		clWaitForEvents(1, &finished_B);
		
		/* Now we will run the same program using only one device. The hi-priority task will have to wait
		until the low priority command-queue has finished */
		printf("\n\n ---Without Device Fission: ---");
		ret = clEnqueueNDRangeKernel(command_queue_LO, kernel_A, 1, NULL, globalWorkSize, localWorkSize,0, NULL, NULL); 
	    ret = clEnqueueNDRangeKernel(command_queue_LO, kernel_B, 1, NULL, globalWorkSize, localWorkSize,0, NULL, NULL); 
	    ret = clEnqueueNDRangeKernel(command_queue_LO, kernel_C, 1, NULL, globalWorkSize, localWorkSize,0, NULL, NULL); 
	}
	else
	{
		//Failed to open the kernel file
		printf("\nFailed to load kernel.");
	}

	//Cleaning 
	ret = clFlush(command_queue_LO);
	ret = clFinish(command_queue_LO);
	ret = clFinish(command_queue_HI);
	ret = clReleaseKernel(kernel_A);
	ret = clReleaseKernel(kernel_B);
	ret = clReleaseKernel(kernel_C);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(inputData_true);
	ret = clReleaseMemObject(inputData_false);
	ret = clReleaseCommandQueue(command_queue_LO);
	ret = clReleaseCommandQueue(command_queue_HI);
	ret = clReleaseContext(context);
	
	scanf("%c",NULL);
}

