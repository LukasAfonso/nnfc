clean:
	rm -rf build

nnfc: src/main.cpp
	mkdir -p build
	g++ -ggdb -Wall -Wextra -o build/main src/main.cpp

all: nnfc