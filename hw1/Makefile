all: ga

ga: ga.cpp
	g++ -o ga -O3 ga.cpp

run: ga
	./ga < cycle.in > cycle.out

clean:
	rm ga

test: ga
	./ga < cycle.in > cycle.out
	./ga < cycle.in.11 > cycle.out.11
	./ga < cycle.in.101 > cycle.out.101
	./ga < cycle.in.21 > cycle.out.21
	./ga < cycle.in.318 > cycle.out.318
	./ga < cycle.in.51 > cycle.out.51
