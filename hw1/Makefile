all: ga

ga: ga.o test_case.o
	g++ -o ga -O3 ga.o test_case.o

ga.o: ga.cpp
	g++ -c -O3 ga.cpp

test_case.o: test_case.h test_case.cpp
	g++ -c -O3 test_case.cpp

run: ga
	./ga < cycle.in > cycle.out

a: ga
	./ga < cycle.in

clean:
	rm -f ga
	rm -f *.o
	rm -f *.out.*

test: ga
	./ga < cycle.in > cycle.out
	./ga < cycle.in.11 > cycle.out.11
	./ga < cycle.in.101 > cycle.out.101
	./ga < cycle.in.21 > cycle.out.21
	./ga < cycle.in.318 > cycle.out.318
	./ga < cycle.in.51 > cycle.out.51
