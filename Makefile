CXX=clang++
FLAGS=-std=c++14 -Wall -Wextra -pedantic -O3

demo: demo.cpp particlefilter.hpp
	$(CXX) $(FLAGS) -o $@ $<

clean:
	rm -f demo
