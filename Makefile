OBJ = main.o cppn.o neat.o util.o extract.o network.o
i=
IMPL = $(addsuffix .o,$(addprefix network-,$(i)))

CFLAGS = -Wall
LDFLAGS = -lm

ifdef IMPL
CFLAGS += -include $(IMPL:.o=.h)
endif

main: $(OBJ) $(IMPL)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(IMPL) $(LDFLAGS)

%.o: %.h
$(OBJ): params.h util.h
neat.o: cppn.h
extract.o: network.h
network.o: cppn.h extract.h
$(IMPL): params.h util.h network.h extract.h

debug: CFLAGS += -g
debug: main

memcheck: valgrind
valgrind: CFLAGS += -O0
valgrind: debug

.PHONY: clean debug memcheck valgrind
clean:
	-rm main *.o

