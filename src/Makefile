INCLUDE = -I/usr/local/include/ -I/usr/include/eigen3
LIBDIR  = -L/usr/local/lib

COMPILERFLAGS = -Wall
CC = g++
CFLAGS = $(COMPILERFLAGS) $(INCLUDE)
LIBRARIES = -lm -std=c++11 -lccd -lfcl

All: fcl

fcl: pybind.cpp simplex.cpp object.cpp
	c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` pybind.cpp simplex.cpp object.cpp -o $@`python3-config --extension-suffix` $(LIBRARIES) $(CFLAGS)

clean:
	rm *.o