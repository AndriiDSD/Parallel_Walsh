
CC=mpicc
CFLAGS= -fopenmp

all:
	$(CC) $(CFLAGS) parallel-walsh.c -o parallel-walsh -lm

clean:
	rm parallel-walsh
	rm input_vector.txt
	rm output_vector.txt
