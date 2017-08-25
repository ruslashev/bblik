all:
	g++ main.cc -lOpenCL -lglut -lGLEW -lGLX -lGL -lGLU -o bblik
	./bblik

smallpt:
	-mv image.ppm prev_image.ppm
	g++ smallpt.cc -O3 -fopenmp -o smallpt
	./smallpt
	gpicview image.ppm &

