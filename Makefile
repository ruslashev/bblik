
all:
	-mv image.ppm prev_image.ppm
	g++ smallpt.cc -O3 -fopenmp -o smallpt
	./smallpt
	gpicview image.ppm &

16:
	-mv image.ppm prev_image.ppm
	g++ smallpt.cc -O3 -fopenmp -o smallpt
	./smallpt 16
	gpicview image.ppm &

32:
	-mv image.ppm prev_image.ppm
	g++ smallpt.cc -O3 -fopenmp -o smallpt
	./smallpt 32
	gpicview image.ppm &

64:
	-mv image.ppm prev_image.ppm
	g++ smallpt.cc -O3 -fopenmp -o smallpt
	./smallpt 64
	gpicview image.ppm &

128:
	-mv image.ppm prev_image.ppm
	g++ smallpt.cc -O3 -fopenmp -o smallpt
	./smallpt 128
	gpicview image.ppm &

200:
	-mv image.ppm prev_image.ppm
	g++ smallpt.cc -O3 -fopenmp -o smallpt
	./smallpt 200
	gpicview image.ppm &

