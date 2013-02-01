OBJ = main.o cppn.o neat.o params.o util.o
DEPS = util.h params.h

CFLAGS = -Wall
LFLAGS = -lm

all: main

main: $(OBJ) $(DEPS)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LFLAGS)

debug: CFLAGS += -g
debug: main

memcheck: valgrind
valgrind: CFLAGS += -O0
valgrind: debug

neat.o: cppn.h

.PHONY: clean
clean:
	-rm main $(OBJ)

