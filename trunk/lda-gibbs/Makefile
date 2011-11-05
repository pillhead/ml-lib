CC = g++ -Wall -O3
LDFLAGS = -lgsl -lm -lgslcblas

GSL_INCLUDE = /usr/include/gsl
GSL_LIB = /usr/lib64

LSOURCE =  tlib.c topicmodel.c
LHEADER =  tlib.h

topicmodel: $(LSOURCE) $(LHEADER)
	$(CC)  -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) -o $@ $(LDFLAGS)

clean:
	rm -f *.o topicmodel