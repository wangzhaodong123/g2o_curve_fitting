INCLUDE_OPENCV = $(shell pkg-config --cflags opencv)
LIBDIR_OPENCV = $(shell pkg-config --libs opencv)

INCLUDE_EIGEN = -I /usr/include/eigen3

LIBDIR_G2O = -L /usr/local/lib/*.so 
INCLUDE_G2O= -I /usr/local/include

all:g2o_curve_fitting

g2o_curve_fitting:g2o_curve_fitting.o
	g++ -std=c++11 -o g2o_curve_fitting g2o_curve_fitting.o $(LIBDIR_OPENCV) $(LIBDIR_G2O)
g2o_curve_fitting.o:g2o_curve_fitting.cpp
	g++ -std=c++11 -c g2o_curve_fitting.cpp $(INCLUDE_IPENCV) $(INCLUDE_EIGEN) $(INCLUDE_G2O)
clean:
	rm -f *.o g2o_curve_fitting
