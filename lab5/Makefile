all: cpufilter

cpufilter: cudappmfilter.o ppmfilter.o readppm.o
	gcc -lglut -lGL -o cpufilter ppmfilter.o readppm.o
	nvcc -lglut -lGL -o cudafilter cudappmfilter.o readppm.o

cudappmfilter.o:
	nvcc -c -o cudappmfilter.o ppmfilter.cu

ppmfilter.o:
	gcc -c ppmfilter.c

readppm.o:
	gcc -c readppm.c

clean:
	rm cpufilter
	rm *.o