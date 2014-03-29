#############################################################################
# Makefile for building: QLZM
#############################################################################

####### Compiler, tools and options
CC            = gcc
CXX           = g++
CFLAGS        = -m64 -pipe -g -Wall -W -D_REENTRANT $(DEFINES)
CXXFLAGS      = -m64 -pipe -g -Wall -W -D_REENTRANT $(DEFINES)
LINK          = g++
LIBS          = -lopencv_highgui -lopencv_core -lopencv_imgproc
AR            = ar cqs
RANLIB        = 
COPY          = cp -f
SED           = sed
COPY_FILE     = $(COPY)
COPY_DIR      = $(COPY) -r
STRIP         = strip
INSTALL_FILE  = install -m 644 -p
INSTALL_DIR   = $(COPY_DIR)
INSTALL_PROGRAM = install -m 755 -p
DEL_FILE      = rm -f
SYMLINK       = ln -f -s
DEL_DIR       = rmdir
MOVE          = mv -f
CHK_DIR_EXISTS= test -d
MKDIR         = mkdir -p

####### Output directory

OBJECTS_DIR   = ./

####### Files

SOURCES       = main.cpp \
		lib_cvip/Image.cpp \
		QLZM.cpp 
		
OBJECTS       = main.o \
		Image.o \
		QLZM.o
		
QMAKE_TARGET  = QLZM
DESTDIR       = 
TARGET        = QLZM

first: all


####### Implicit rules

.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules

all: Makefile $(TARGET)

$(TARGET):  $(OBJECTS)  
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(OBJCOMP) $(LIBS) && $(DEL_FILE) main.o QLZM.o Image.o
	{ test -n "$(DESTDIR)" && DESTDIR="$(DESTDIR)" || DESTDIR=.; } && test $$(gdb --version | sed -e 's,[^0-9]\+\([0-9]\)\.\([0-9]\).*,\1\2,;q') -gt 72 && gdb --nx --batch --quiet -ex 'set confirm off' -ex "save gdb-index $$DESTDIR" -ex quit '$(TARGET)' && test -f $(TARGET).gdb-index && objcopy --add-section '.gdb_index=$(TARGET).gdb-index' --set-section-flags '.gdb_index=readonly' '$(TARGET)' '$(TARGET)' && rm -f $(TARGET).gdb-index && rm *.o || true

####### Compile

main.o:  main.cpp lib_cvip/Image.h \
		lib_cvip/Definitions.h \
		QLZM.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o main.o main.cpp

Image.o: lib_cvip/Image.cpp lib_cvip/Image.h \
		lib_cvip/Definitions.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o Image.o lib_cvip/Image.cpp

QLZM.o: QLZM.cpp lib_cvip/Image.h \
		lib_cvip/Definitions.h \
		QLZM.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o QLZM.o QLZM.cpp

