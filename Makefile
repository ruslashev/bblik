
all:
	g++ smallpt.cc -O3 -fopenmp -o smallpt
	./smallpt
	gpicview image.ppm

16:
	g++ smallpt.cc -O3 -fopenmp -o smallpt
	./smallpt 16
	gpicview image.ppm

