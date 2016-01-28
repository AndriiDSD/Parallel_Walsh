/*
* Authors: Andrii Hlyvko, Nadiia Chepurko, Lucas Rivera
* Usage: ./parallel-walsh <input size> <implementation>
* This program computes the Walsh-Hadamard transform of an array of random integers.
* There are two implementations available. 1 is for the simple vector multiplication and
* 2 is for the fast implementation. The output will be saved in files in the 
* current folder. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <errno.h>
#include <stdint.h>
#include <time.h>
//#include "timing.h"
#include "timer.h"
#ifdef _OPENMP
#include <omp.h>
#endif

unsigned int  seed; 


/*
* This function counts the number of bits in a 32 bit integer
*/
int number_of_set_bits(uint32_t i)
{
	i = i - ((i >> 1) & 0x55555555);
	i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
	return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

/*
* This function checks if a number is a power of two.
*/
int is_power_two(int x)
{
	return x && !(x & (x - 1));
}

/*
* This function computes a modulo b
*/
int mod(int a, int b)
{
	int m=(((a%b)+b)%b);
	return m;
}

/*
* This function adds two vectors in parallel using openmp
*/
void vector_add(int *a,int *b, int size)
{
	int i=0;
	#pragma omp parallel for default(none) private(i) shared(size, a, b)
	for(i=0; i<size; i++)
	{
		a[i] = a[i] + b[i];
	}
}

/*
* This function computes b-a and stores the result in a
*/
void vector_sub(int *a,int *b, int size)
{
	int i=0;
	#pragma omp parallel for default(none) private(i) shared(size, a, b)
	for(i=0; i<size; i++)
	{
		a[i] = b[i] - a[i];
	}
}

/*
* This is the sequential implementation of the simple hadamard transform.
* It multiplies the input vector by the hadamard matrix.
*/
void simple_sequential_walsh(int *input,int *out, int size)
{
	int i=0,j=0;
	for(i=0; i<size; i++)
	{
		out[i] = 0;
		for(j=0; j < size;j++)
		{	
			int factor =1;
			if(number_of_set_bits(i&j)%2)
				factor=-1;
			else
				factor=1;
			out[i]+=input[j]*factor;
		}
	}
}

/*
* This is the sequential implementation of the fast hadamard transform
*/
void fast_sequential_walsh(int *input, int size)
{
	int lSize = log2(size);
	int i=0,j=0,k=0;
	int a=0,b=0,tmp=0;
	for(i=0; i< lSize;i++)
	{
		for(j=0;j<(1 << lSize); j += 1 << (i+1))
		{
			for(k=0; k < (1 << i); k++)
			{
				a=input[j+k];
				b=input[j+k+(1<<i)];
				tmp=a;
				a+=b;
				b=tmp-b;
				input[j+k]=a;
				input[j+k+(1<<i)]=b;
			}
		}
	}
}

/*
* This function is used within a node to generate the input randomly.
* Within a node openmp is used to further break up the input generation.
*/
void generate_input_mpi(int *input, int size, int my_rank, int comm_sz, MPI_Comm comm)
{	
	int local_size = size / comm_sz;
	if(local_size==0)
		local_size++;
	int local_start = my_rank*local_size;
	int local_end = local_start + local_size;

	int *local_input = (int *)malloc(local_size*sizeof(int));

	int i=0, id=my_rank;

	srand(time(NULL) +my_rank);
	if(my_rank < size)
	{
		#pragma omp parallel for private(i,id) shared(local_size)
			for(i=0; i<local_size; i++)
			{
				//id=my_rank*omp_get_num_threads() +omp_get_thread_num();
				local_input[i] = ((unsigned int)(rand_r((unsigned int *)&id))+rand())%11;
			}
			
	}
	MPI_Allgather(local_input, local_size, MPI_INT, input, local_size, MPI_INT,comm);

	if((local_size*comm_sz) < size)
	{
		if(my_rank == 0)
		{
			#pragma omp parallel for private(i,id) shared(local_size)
			for(i=(local_size*comm_sz); i<size; i++)
			{
				//id=my_rank*omp_get_num_threads() +omp_get_thread_num();
				input[i] = ((unsigned int)(rand_r((unsigned int *)&id))+rand())%11;
			}
			
		}
		MPI_Bcast((input+(local_size*comm_sz)), (size-(local_size*comm_sz)), MPI_INT, 0, comm);
	}


	free(local_input);
	
}

/*
* This is the parallel implementation of the simple hadamard transform.
* The outer loop has a loop carried dependency so it was not parallelized with openmp
* First the nodes divide the input between themselves and then on a node the input is 
* computed in parallel. No communication is needed for this implementation except the final
* gather of the vector.
*/
void simple_parallel_walsh(int* vector, int size, int my_rank, int comm_sz, MPI_Comm comm)
{
	

	int local_size = size / comm_sz;
	if(local_size == 0)
		local_size++;

	int local_start = my_rank*local_size;
	int local_end = local_start + local_size;

	int *local_output = (int *)malloc(local_size*sizeof(int));
	bzero(local_output,local_size*sizeof(int));
	int i=0,j=0,factor=1;
	if(my_rank <size)
	{
		#pragma omp parallel for default(none) private(j,factor) shared(size,vector,local_start,local_size,local_end,local_output,i)
			for(i=local_start; i<local_end; i++)
			{
				//local_output[i%local_size] = 0;
				for(j=0; j < size;j++)
				{	
					if(number_of_set_bits(i&j)%2)
						factor=-1;
					else
						factor=1;
					local_output[i%local_size]+=vector[j]*factor;
				}
			}
		
	}

	if((local_size*comm_sz) < size)
	{
		if(my_rank == 0)
		{
			//bzero(local_output,local_size*sizeof(int));
			#pragma omp parallel for default(none) private(j,factor) shared(comm_sz,size,vector,local_start,local_size,local_end,local_output,i)
			for(i=local_size*comm_sz; i<size; i++)
			{
				local_output[i-local_size*comm_sz] = 0;
				for(j=0; j < size;j++)
				{	
					if(number_of_set_bits(i&j)%2)
						factor=-1;
					else
						factor=1;
					local_output[i-local_size*comm_sz]+=vector[j]*factor;
				}
			}
			memcpy(vector+local_size*comm_sz, local_output,(size-local_size*comm_sz)*sizeof(int));
		}
	}

	MPI_Gather(local_output, local_size, MPI_INT, vector, local_size, MPI_INT, 0, MPI_COMM_WORLD);

	

	free(local_output);
}

/*
* This is the openmp version of the hadamard transform. It will be called within a node to compute the
* hadamerd transform of the local input.
*/
void fast_walsh_omp(int *input, int size)
{
	if(size==0)
		return;
	int lSize = log2(size);
	int i=0,j=0,k=0;
	int a=0,b=0,tmp=0;
	for(i=0; i< lSize;i++)
	{
		#pragma omp parallel for default(none) shared(lSize, i, input) private(j,k, a, b, tmp)
		for(j=0;j<(1 << lSize); j += 1 << (i+1))
		{
			for(k=0; k < (1 << i); k++)
			{
				a=input[j+k];
				b=input[j+k+(1<<i)];
				tmp=a;
				a+=b;
				b=tmp-b;
				input[j+k]=a;
				input[j+k+(1<<i)]=b;
			}
		}
	}
}

/*
* This is the parallel implementation of the fast hadamard transform. 
* First each node computes the local hadamard transform using openmp. Then each node needs 
* to communicate its local result to other nodes. If the result is send forward it will be subtracted, else it 
* will be added.
*/
void fast_parallel_walsh(int* vector, int size, int my_rank, int comm_sz, MPI_Comm comm)
{
	int local_size = size / comm_sz;
	if(local_size == 0)
		local_size++;
	
	int local_start = my_rank*local_size;
	int local_end = local_start + local_size;
	int *local_input = (int *)malloc(local_size*sizeof(int));
	memcpy(local_input, (vector+local_start), local_size*sizeof(int));
	fast_walsh_omp(local_input,local_size);

	int *rcv = (int *)malloc(local_size*sizeof(int));

	int i=0,j=0,k=0;
	int dest=0;
	int commLog = log2(comm_sz);
	for(i=0; i < commLog; i++)
	{
		if(my_rank % 2 == 0 )
			dest = mod((my_rank + (1 << i)) , (1 << (i+1))) + ((my_rank/( 1 << (i+1) ) )*(1 << (i+1)) );
		else
			dest = mod((my_rank - (1 << i)) , (1 << (i+1))) + ((my_rank/( 1 << (i+1) ) )*(1 << (i+1)) );
		
		if(my_rank > dest)
		{
			MPI_Send(local_input, local_size, MPI_INT, dest, 0, comm);
			MPI_Recv(rcv, local_size, MPI_INT, dest, 0, comm, MPI_STATUS_IGNORE);	
		}
		else
		{
			MPI_Recv(rcv, local_size, MPI_INT, dest, 0, comm, MPI_STATUS_IGNORE);
			MPI_Send(local_input, local_size, MPI_INT, dest, 0, comm);
		}


		if(dest > my_rank)
			vector_add(local_input, rcv, local_size);
		else
			vector_sub(local_input, rcv, local_size);
		
	}


	MPI_Gather(local_input, local_size, MPI_INT, vector, local_size, MPI_INT, 0, comm);

	free(local_input);
	free(rcv);
}


/*
* The main function computes the hadamard transform of a randomly generated input vector
* and saves the result to a file.
*/
int main(int argc, char *argv[])
{
	// check the number of arguments
	if(argc != 3)
	{
		perror("Usage: ./parallel-walsh <size> <implementation: 1 for simple, 2 for fast>\n");
		exit(EXIT_FAILURE);
	}
	int size=0, implementation = 0;
	
	// convert the arguments to integers
	size = strtol(argv[1], NULL, 10);
	implementation = strtol(argv[2], NULL, 10);
	if(errno == EINVAL)
	{
		perror("Error converting program arguments to integers\n");
		exit(EXIT_FAILURE);
	}

	// the size of the input vector has to be a power of two
	if(!is_power_two(size))
	{
		perror("Error. Size is not power of two\n");
		exit(EXIT_FAILURE);
	}
	
	// check for the impelmentation version
	if(implementation != 1 && implementation != 2)
	{
		perror("Error. Invalid implementation selection\n");
		exit(EXIT_FAILURE);
	}

	seed =(unsigned int) time(NULL);  // generate a seed for random generator

	double t=0.0,tS=0.0;
	int my_rank=0,comm_sz=1;
	//int *out_sequential;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);


	int *input = (int *) malloc(size*sizeof(int));
	
	// generate the input vector in parallel
	generate_input_mpi(input, size, my_rank, comm_sz, MPI_COMM_WORLD);

	// process 0 will save the input vector in a file
	if(my_rank == 0)
	{
		FILE *input_f= fopen("input_vector.txt", "w+");

		fprintf(input_f, "Randomly generated input vector:\n");
		fprintf(input_f,"i                vector[i]\n");
		int i=0;

		for(i=0;i<size;i++)
		{
			fprintf(input_f,"%d             %5d\n",i,input[i]);
		}
		fclose(input_f);

		// a vector that holds the result of sequential hadamard transform
		//out_sequential = (int *)malloc(size*sizeof(int));
	}

	

	
	if(implementation == 1)
	{
		// compute the sequential and parallel simple hadamard transform
		//if(my_rank == 0)
		//{
			// process 0 computes the sequential simple hadamard transform
		//	tS=get_time();
		//	simple_sequential_walsh(input, out_sequential, size);
		//	tS=get_time()-tS;
		//	printf("Time of simple sequential walsh was:%3.5f seconds\n",tS);
		//}

		t=get_time();
		simple_parallel_walsh(input, size, my_rank, comm_sz, MPI_COMM_WORLD);
		t=get_time()-t;
		
		// output the timing 
		if(my_rank==0)
		{
			printf("Time of simple parallel walsh was:%3.5f seconds\n",t);
			//printf("The speedup is: %3.3f\n",(((double)tS)/t));
		}

	}
	else
	{
		// compute the sequential and parallel fast hadamard transform
		//if(my_rank == 0)
		//{
		//	memcpy(out_sequential, input, size*sizeof(int));
		//	tS=get_time();
		//	fast_sequential_walsh(out_sequential, size);
		//	tS=get_time()-tS;	
		//}

		t=get_time();
		fast_parallel_walsh(input, size, my_rank, comm_sz, MPI_COMM_WORLD);
		t=get_time()-t;

		if(my_rank==0)
		{	
			//printf("Time of fast sequential walsh was:%3.5f seconds\n",tS);
			printf("Time of fast parallel walsh was:%3.5f seconds\n",t);
			//printf("The speedup is: %3.3f\n",(((double)tS)/t));
		}

	}
	
	

	if(my_rank == 0)
	{
		// process 0 saves the result to a file
		FILE *output_f= fopen("output_vector.txt", "w+");

		fprintf(output_f, "Output vector:\n");
		//fprintf(output_f,"i             output_parallel[i]     output_sequential[i]\n");
		fprintf(output_f,"i             output_parallel[i]\n");
		int i=0;

		for(i=0;i<size;i++)
		{
		//	fprintf(output_f,"%d             %5d                       %5d\n",i,input[i],out_sequential[i]);
			fprintf(output_f,"%d             %5d\n",i,input[i]);
		}
		fclose(output_f);
		//free(out_sequential);
	}
	
	free(input);
	MPI_Finalize();
	return 0;
}



