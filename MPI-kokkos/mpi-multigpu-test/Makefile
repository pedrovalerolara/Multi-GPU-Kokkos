#KOKKOS_PATH = ${HOME}/Kokkos/
KOKKOS_PATH = ${HOME}/kokkos-multigpu-backend/kokkos-master/
KOKKOS_DEVICES = "OpenMP"
EXE_NAME = "MULTIGPU"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"


ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
#CXX = mpicxx
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper  -I/sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/xl-16.1.1-10/spectrum-mpi-10.4.0.3-20210112-dzedzfvocsuzkm4jkqe7o64x53yhq7nm/include -L/sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/xl-16.1.1-10/spectrum-mpi-10.4.0.3-20210112-dzedzfvocsuzkm4jkqe7o64x53yhq7nm/lib -lmpiprofilesupport -lmpi_ibm -Xcompiler -fopenmp  
#CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper -Xcompiler -fopenmp
CXXFLAGS = -O3 -I/sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/xl-16.1.1-10/spectrum-mpi-10.4.0.3-20210112-dzedzfvocsuzkm4jkqe7o64x53yhq7nm/include
LINKFLAGS =  -lmpiprofilesupport -lmpi_ibm 
EXE = ${EXE_NAME}.cuda
#KOKKOS_ARCH = "Pascal60"
KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
CXX = g++
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = "Power9"
endif

CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)
