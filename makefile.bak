CXX=g++
RM=rm -f

GKINCHOME=/home/karypisg/msharma/dev/George/trunk 
GKLIBHOME=/home/karypisg/msharma/dev/George/trunk/build/Linux-x86_64/ 

EIGENPATH=/home/karypisg/msharma/dev/eigen
SPECTRAPATH=/home/karypisg/msharma/dev/spectra/include
SVDLIBPATH=/home/karypisg/msharma/dev/SVDLIBC

#Standard Libraries
STDLIBS=-lm -lpthread -fopenmp

#external libraries
EXT_LIBS=-lGKlib -lsvd 
<<<<<<< Updated upstream
EXT_LIBS_DIR=-L$(GKLIBHOME) -L$(SVDLIBPATH) 

CPPFLAGS=-g -O3 -Wall -fopenmp -std=c++11 -I$(GKINCHOME) -I$(SVDLIBPATH)
LDFLAGS=-g
LDLIBS=$(STDLIBS) $(EXT_LIBS_DIR) $(EXT_LIBS) 

SRCS=model.cpp modelMF.cpp io.cpp longTail.cpp util.cpp svdFrmsvdlib.cpp topBucketComp.cpp confCompute.cpp main.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: mf

mf: $(OBJS)
	$(CXX) $(LDFLAGS) -o mf $(OBJS) $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CPPFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS)

dist-clean: clean
	$(RM) tool

include .depend

