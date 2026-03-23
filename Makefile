clean:
	rm -rf build

nnfc: src/main.cpp
	mkdir -p build
	g++ -o build/main src/main.cpp

all: nnfc