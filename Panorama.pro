TEMPLATE = app
CONFIG += console
CONFIG -= qt

INCLUDEPATH += /usr/local/opencv/include/
LIBS += -L/usr/local/opencv/lib -lopencv_nonfree -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lopencv_flann -lopencv_calib3d

SOURCES += main.cpp

