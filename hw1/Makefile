all: ga

ga: ga.o test_case.o
	g++ -O3 -pg -o ga ga.o test_case.o
	ctags *

ga.o: ga.cpp
	g++ -O3 -c ga.cpp

test_case.o: test_case.h test_case.cpp
	g++ -O3 -c test_case.cpp

run: ga
	./ga < cycle.in > cycle.out

clean:
	rm -f ga
	rm -f *.o
	rm -f *.out
	rm -f *.out.*
	rm -f core
