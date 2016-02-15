PROJ = matrix_mul
CC = g++
NVCC = nvcc -Xcompiler -fopenmp --ptxas-options=-v

CFLAGS = -c -g -o3 -Wall -I/opt/local/include -I$(HOME)/cppunit/include  -fopenmp
LDFLAGS = -L/opt/local/lib -L$(HOME)/cppunit/lib
LIBS = -lcppunit -ldl -lgomp
OBJS = matrix_mul.o tests.o

all: $(PROJ)

$(PROJ): $(OBJS) $(TEST_CASES)
	$(NVCC) $(LDFLAGS) $^ -o $@ $(LIBS)

%.o : %.cu %.h
	$(NVCC) -c $< -o $@

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@ 

clean:
	rm -f $(PROJ) $(OBJS) $(TEST_CASES) *.xml
