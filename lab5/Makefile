all: cpufilter

LDFLAGS=-lglut -lGL

cpufilter: cudappmfilter.o ppmfilter.o readppm.o
	gcc -o cpufilter ppmfilter.o readppm.o $(LDFLAGS)
	nvcc -o cudafilter cudappmfilter.o readppm.o $(LDFLAGS)

cudappmfilter.o:
	nvcc -c -o cudappmfilter.o ppmfilter.cu

ppmfilter.o:
	gcc -c ppmfilter.c

readppm.o:
	gcc -c readppm.c

clean:
	rm cpufilter
	rm *.o