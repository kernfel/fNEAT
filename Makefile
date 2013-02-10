OBJ = main.o cppn.o neat.o params.o util.o extract.o network.o

CFLAGS = -Wall
LDFLAGS = -lm

all: main

main: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

debug: CFLAGS += -g
debug: main

memcheck: valgrind
valgrind: CFLAGS += -O0
valgrind: debug

.PHONY: clean
clean:
	-rm main $(OBJ)

