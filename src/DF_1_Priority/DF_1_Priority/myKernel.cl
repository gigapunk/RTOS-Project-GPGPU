__kernel void lowPriorityTask()
{
	int workID = get_global_id(0);
	
	int n = INT_MAX, first = 0, second = 1, next, c;
 
	printf("\nI'm a [low ] priority task and i'm occupying the LO-Device\t (LO-LO)") ;
    for ( c = 0 ; c < n ; c++ )
   {
      if ( c <= 1 )
         next = c;
      else
      {
         next = first + second;
         first = second;
         second = next;
      }
    }
    
}

__kernel void highPriorityTask(__global int* data)
{
	int workID = get_global_id(0);
	printf("\nI'm a [high] priority task ");
	data[workID] == 1 ? printf("and I'm running on the HI-device\t (HI-HI)") : printf("but I'm running on the LO-device\t (LO-LO)");
}