all: ga

ga: ga.o test_case.o lk.o edgetour.o tsplib_io.o cpputil.o
	g++ -O3 -o ga ga.o test_case.o lk.o edgetour.o tsplib_io.o cpputil.o

ga.o: ga.cpp
	g++ -O3 -c ga.cpp

test_case.o: test_case.h test_case.cpp
	g++ -O3 -c test_case.cpp

lk.o: lk.h lk.cc
	g++ -O3 -c lk.cc -DDISTANCE_CALC_METHOD=1

edgetour.o: edgetour.cc
	g++ -O3 -c edgetour.cc

tsplib_io.o: tsplib_io.h tsplib_io.cc
	g++ -O3 -c tsplib_io.cc -DDISTANCE_CALC_METHOD=1

cpputil.o: cpputil.h cpputil.cc
	g++ -O3 -c cpputil.cc

run: ga
	./ga < cycle.in.200 > cycle.out

clean:
	rm -f ga
	rm -f *.o
	rm -f *.out
	rm -f *.out.*
	rm -f core
	rm -f tags
