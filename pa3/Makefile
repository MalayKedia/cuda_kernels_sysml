NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -std=c++11 -lcublas -Isrc

SRC_DIR = src
BIN_DIR = bin

# Shared kernels linked into every variant.
COMMON = $(SRC_DIR)/matmul.cu $(SRC_DIR)/relu.cu

# One binary per forward-pass variant.
VARIANTS = $(wildcard $(SRC_DIR)/ffnn_*.cu)
TARGETS = $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%, $(VARIANTS))

all: $(BIN_DIR) $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: $(SRC_DIR)/%.cu $(COMMON) main.cc force | $(BIN_DIR)
	@$(NVCC) -DNOT_LEETGPU $(NVCC_FLAGS) -o $@ $< $(COMMON) main.cc
	@echo "Running $@..."
	$@
	@echo ""

clean:
	rm -f $(BIN_DIR)/*

force:

.PHONY: all clean force
