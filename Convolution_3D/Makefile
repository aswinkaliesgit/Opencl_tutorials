ifndef CPPC
	CPPC = g++
endif

CPP_COMMON = ../Cpp_common

CCFLAGS = -std=c++11

INC = -I$(CPP_COMMON) $(shell pkg-config --cflags opencv4)

LIBS = -lOpenCL -lrt $(shell pkg-config --libs opencv4)

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	CCFLAGS += -stdlib=libc++
	LIBS = -framework OpenCL $(shell pkg-config --libs opencv4)
endif

CCFLAGS += -DDEVICE=$(DEVICE)

Convolution_3D: Convolution_3D.cpp
	$(CPPC) $^ $(INC) $(CCFLAGS) $(LIBS) -o $@

clean:
	rm -f Convolution_3D.cpp
	
