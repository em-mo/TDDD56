all: lenna

LDFLAGS=-lglut -lGL -lOpenCL

lenna: cl3.o CLutilities.o readppm.o 3.cl
	gcc -o lenna cl3.o CLutilities.o readppm.o $(LDFLAGS)

cl3.o: cl3.c
	gcc -c cl3.c

readppm.o: readppm.c
	gcc -c readppm.c

CLutilities.o: CLutilities.c
	gcc -c CLutilities.c

clean:
	rm cpufilter
	rm cudafilter
	rm *.o