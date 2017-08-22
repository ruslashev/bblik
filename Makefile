
all:
	-mv image.ppm prev_image.ppm
	g++ smallpt.cc -O3 -fopenmp -o smallpt
	./smallpt
	gpicview image.ppm

16:
	-mv image.ppm prev_image.ppm
	g++ smallpt.cc -O3 -fopenmp -o smallpt
	./smallpt 16
	gpicview image.ppm

