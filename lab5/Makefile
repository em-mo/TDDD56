all: cpufilter

LDFLAGS=-lglut -lGL
NFLAGS=-Xptxas -v

cpufilter: cudappmfilter.o ppmfilter.o readppm.o
	gcc -o cpufilter ppmfilter.o readppm.o $(LDFLAGS)
	nvcc -o cudafilter cudappmfilter.o readppm.o $(LDFLAGS)

cudappmfilter.o: ppmfilter.cu
	nvcc $(NFLAGS) -c -o cudappmfilter.o ppmfilter.cu

ppmfilter.o: ppmfilter.c
	gcc -c ppmfilter.c

readppm.o: readppm.c
	gcc -c readppm.c

clean:
	rm cpufilter
	rm cudafilter
	rm *.o