all:
	g++ main.cc screen.cc ogl.cc -lOpenCL -lSDL2 -lGLEW -lGLX -lGL -o bblik
	./bblik

smallpt:
	-mv image.ppm prev_image.ppm
	g++ smallpt.cc -O3 -fopenmp -o smallpt
	./smallpt
	gpicview image.ppm &

