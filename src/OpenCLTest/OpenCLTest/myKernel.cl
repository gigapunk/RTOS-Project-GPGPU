__kernel void square(__global int* number, __global int* result)
{
	int workID = get_global_id(0);
	printf("Work number: %d Input: %d \n",workID,number[workID]) ;
	result[workID] = number[workID] * number[workID];
}