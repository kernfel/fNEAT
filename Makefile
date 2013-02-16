OBJ = main.o cppn.o neat.o util.o extract.o network.o robot-simplistic.o
i=

ifdef i
eNETWORK = $(addsuffix .o,$(addprefix network-,$(i)))
else
eNETWORK = network-static.o
endif

CFLAGS = -Wall -include $(eNETWORK:.o=.h)
LDFLAGS = -lm

main: $(OBJ) $(eNETWORK)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(eNETWORK) $(LDFLAGS)

%.o: %.h
$(OBJ): params.h util.h
neat.o: cppn.h
extract.o: network.h
network.o: cppn.h extract.h
$(eNETWORK): params.h util.h network.h extract.h

debug: CFLAGS += -g
debug: main

memcheck: valgrind
valgrind: CFLAGS += -O0
valgrind: debug

.PHONY: clean debug memcheck valgrind
clean:
	-rm main *.o

