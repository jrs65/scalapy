

all: bctest

bctest: bctest.c bcutil.c
	gcc -o bctest bcutil.c bctest.c 

