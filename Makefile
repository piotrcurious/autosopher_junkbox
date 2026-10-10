CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2
LDFLAGS = -lm

OBJS = main.o forward_learning_mod.o noise_transcendence_mod.o
TARGETS = autosopher_suite forward_learning noise_and_transcendence

all: $(TARGETS)

autosopher_suite: $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

forward_learning: forward_learning.o forward_learning_mod.o noise_transcendence_mod.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

noise_and_transcendence: noise_and_transcendence.o forward_learning_mod.o noise_transcendence_mod.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGETS)

.PHONY: all clean
