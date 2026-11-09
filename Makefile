CXX = g++
CPPFLAGS ?=
# AVX2/FMA are required. Avoid fast-math; no host-specific march is needed.
CXXFLAGS ?= -std=c++20 -O3 -g -fno-omit-frame-pointer -Wall -Wextra -mavx2 -mfma -pthread
LDFLAGS ?=
LDLIBS ?=
EXEC = spectral_tracer
OBJ = main.o tests/geometry_test.o
DEP = $(OBJ:.o=.d)

.PHONY: all clean test
all: $(EXEC)

$(EXEC): main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

tests/geometry_test: tests/geometry_test.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

test: $(EXEC) tests/geometry_test
	./tests/geometry_test
	python3 tests/smoke.py

clean:
	rm -f $(OBJ) $(DEP) $(EXEC) tests/geometry_test

-include $(DEP)
