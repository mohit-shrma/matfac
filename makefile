CXX=g++
RM=rm -f

GKINCHOME=/home/grad02/mohit/George/GKlib/trunk
GKLIBHOME=/home/grad02/mohit/George/GKlib/trunk/build/Linux-x86_64/

#Standard Libraries
STDLIBS=-lm -lpthread

#external libraries
EXT_LIBS=-lGKlib 
EXT_LIBS_DIR=-L$(GKLIBHOME) 

CPPFLAGS=-g -o3 -Wall -std=c++11 -I$(GKINCHOME)
LDFLAGS=-g
LDLIBS=$(STDLIBS) $(EXT_LIBS_DIR) $(EXT_LIBS) 

SRCS=model.cpp modelMF.cpp modelMFWtRegArb.cpp  io.cpp util.cpp main.cpp
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

