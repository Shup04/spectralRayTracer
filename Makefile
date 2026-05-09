CXX = g++
#CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -mavx2 -mfma -march=native -pthread
CXXFLAGS = -std=c++20 -O3 -g -fno-omit-frame-pointer -Wall -Wextra -mavx2 -mfma -march=native -pthread
# Add -DPROFILE_SYMBOLS while using perf if inlined hot functions collapse into render_sample_tiles.
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
