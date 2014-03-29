#-------------------------------------------------
#
# Project created by QtCreator 2012-11-19T09:41:59
#
#-------------------------------------------------

QT       += core
QT       -= gui

TARGET = QLZM
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

HEADERS     += ./lib_cvip/Image.h \
               ./lib_cvip/Definitions.h \
                QLZM.h

SOURCES     += main.cpp \
               ./lib_cvip/Image.cpp \
               QLZM.cpp

INCLUDEPATH += /usr/include/opencv2 /usr/local/include/opencv2
LIBS += -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_video

#QMAKE_CXXFLAGS += -fopenmp -O3
