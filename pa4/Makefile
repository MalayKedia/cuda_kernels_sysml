NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -std=c++11 -Isrc

SRC_DIR = src
BIN_DIR = bin

TARGET = $(BIN_DIR)/flash_attention

all: $(BIN_DIR) $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(SRC_DIR)/flash_attention.cu main.cc force | $(BIN_DIR)
	@$(NVCC) $(NVCC_FLAGS) -o $@ $(SRC_DIR)/flash_attention.cu main.cc
	@echo "Running $@..."
	$@
	@echo ""

# Parts A-C: PyTorch reference, FlashAttention in Python loops, built-in fused kernel.
torch:
	python3 attention.py

# Occupancy / shared memory / global throughput for the Part D kernel.
profile: $(TARGET)
	ncu --set full $(TARGET)

clean:
	rm -f $(BIN_DIR)/*

force:

.PHONY: all clean torch profile force
