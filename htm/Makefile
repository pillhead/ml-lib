CC = g++ -Wall -O2 -pg
LDFLAGS = -lgsl -lm -lgslcblas

GSL_INCLUDE = /usr/include/gsl/
GSL_LIB = /usr/lib/

LSOURCE =  utils.cpp corpus.cpp state.cpp hdp.cpp main.cpp
LHEADER =  utils.h corpus.h hdp.h state.h Timer.h

htm: $(LSOURCE) $(HEADER)
	$(CC)  -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) -o $@ $(LDFLAGS)

clean:
	rm -f *.o htm
