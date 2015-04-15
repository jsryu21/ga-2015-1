all: ga

ga: ga.o test_case.o
	#g++ -g -o ga ga.o test_case.o
	g++ -O3 -pg -o ga ga.o test_case.o
	ctags *

ga.o: ga.cpp
	#g++ -g -c ga.cpp
	g++ -O3 -pg -c ga.cpp

test_case.o: test_case.h test_case.cpp
	#g++ -g -c test_case.cpp
	g++ -O3 -pg -c test_case.cpp

run: ga
	./ga < cycle.in > cycle.out

a: ga
	./ga < cycle.in.11

aa: ga
	./ga < cycle.in.101

clean:
	rm -f ga
	rm -f *.o
	rm -f *.out.*
	rm -f core
	rm -f *profile

test: ga
	./ga < cycle.in > cycle.out
	./ga < cycle.in.11 > cycle.out.11
	./ga < cycle.in.101 > cycle.out.101
	./ga < cycle.in.21 > cycle.out.21
	./ga < cycle.in.318 > cycle.out.318
	./ga < cycle.in.51 > cycle.out.51

p: ga
	./ga 1 1 1 1 < cycle.in.11
	gprof ga gmon.out > ga_profile
