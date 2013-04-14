__kernel void Kernel_A(__global int* input, __global int* output)
{
	printf("\nExecuting NODE A..") ;
}

__kernel void Kernel_B(__global int* input, __global int* output)
{
	printf("\nExecuting NODE B..") ;
}


__kernel void Kernel_C(__global int* input, __global int* output)
{
	printf("\nExecuting NODE C..") ;
}

__kernel void Kernel_D(__global int* input1,__global int* input2, __global int* output)
{
	printf("\nExecuting NODE D..") ;
}


