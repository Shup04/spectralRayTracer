CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -mavx2 -mfma -march=native

# We explicitly do NOT use -ffast-math to guarantee deterministic floating-point execution.

SRC = main.cpp
OBJ = $(SRC:.cpp=.o)
EXEC = spectral_tracer

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)
