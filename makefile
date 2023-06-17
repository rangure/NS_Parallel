MAKE   = make
TARGET = nsexe
SOURCE = Navier_Stokes.cpp

default:
	mpicxx -std=c++11 -o $(TARGET) $(SOURCE)
