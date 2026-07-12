CC ?= cc
CFLAGS ?= -O3 -march=native -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS += -Iinclude
CFLAGS += -pthread
LDLIBS += -pthread
OMP ?= 1
ifeq ($(OMP),1)
CFLAGS += -fopenmp
LDLIBS += -fopenmp
else
CFLAGS += -Wno-unknown-pragmas
endif
MODEL ?= nuscenes_multihead.ppw
CHECKPOINT ?= ckpts/pp_multihead_nds5823_updated.pth
CONFIG ?= cfgs/pointpillars.yaml
NVCC ?= /usr/local/cuda-12.4/bin/nvcc
CUDA_ARCH ?= sm_89

.PHONY: all model cuda test portable-test prepare-data checkpoint-oracle evaluate clean
all: build/pointpillars
cuda: build/pointpillars_cuda

model: $(MODEL)
$(MODEL): tools/export_checkpoint.py $(CHECKPOINT) $(CONFIG)
	python3 tools/export_checkpoint.py $(CHECKPOINT) $(CONFIG) $@

build/pointpillars: src/main.c src/model.c src/voxel.c src/infer_cpu.c src/decode.c src/tui.c include/pp_model.h include/pointpillars.h include/pp_infer.h include/pp_decode.h include/pp_tui.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/main.c src/model.c src/voxel.c src/infer_cpu.c src/decode.c src/tui.c -lm $(LDLIBS) -o $@

build/pointpillars_cuda: src/main.c src/model.c src/voxel.c src/infer_cpu.c src/decode.c src/tui.c src/infer_cuda.cu include/pp_cuda.h
	mkdir -p build/cuda
	$(CC) $(CPPFLAGS) $(CFLAGS) -DPP_WITH_CUDA -c src/main.c -o build/cuda/main.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/model.c -o build/cuda/model.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/voxel.c -o build/cuda/voxel.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/infer_cpu.c -o build/cuda/infer_cpu.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/decode.c -o build/cuda/decode.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/tui.c -o build/cuda/tui.o
	$(NVCC) $(CPPFLAGS) -O3 -arch=$(CUDA_ARCH) -c src/infer_cuda.cu -o build/cuda/infer_cuda.o
	$(NVCC) -Xcompiler -fopenmp -Xcompiler -pthread build/cuda/*.o -lgomp -lm -o $@

build/test_model: src/model.c tests/test_model.c include/pp_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/model.c tests/test_model.c -o $@

build/test_voxel: src/voxel.c tests/test_voxel.c include/pointpillars.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/voxel.c tests/test_voxel.c -lm -o $@

build/test_decode: src/decode.c src/infer_cpu.c src/model.c tests/test_decode.c include/pp_decode.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/decode.c src/infer_cpu.c src/model.c tests/test_decode.c -lm $(LDLIBS) -o $@

build/test_tui: src/tui.c tests/test_tui.c include/pp_tui.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/tui.c tests/test_tui.c -lm $(LDLIBS) -o $@

test: model build/pointpillars build/test_model build/test_voxel build/test_decode build/test_tui
	./build/test_model $(MODEL)
	./build/test_voxel
	./build/test_decode
	./build/test_tui

portable-test:
	$(MAKE) clean
	$(MAKE) OMP=0 test

prepare-data:
	python3 tools/prepare_nuscenes.py --root /data/nuscenes --output /data/nuscenes/pointpillars_10sweep

checkpoint-oracle: test
	@f=$$(find /data/nuscenes/pointpillars_10sweep -name '*.bin' | head -1); \
	OMP_NUM_THREADS=16 ./build/pointpillars infer $(MODEL) "$$f" /tmp/nuscenes_cpu.ppout 5; \
	python3 tools/oracle_checkpoint.py $(CHECKPOINT) "$$f" /tmp/nuscenes_cpu.ppout

evaluate: cuda
	rm -rf build/nuscenes-detections
	./build/pointpillars_cuda batch-cuda $(MODEL) /data/nuscenes/pointpillars_10sweep build/nuscenes-detections
	mkdir -p evaluation
	python3 tools/make_submission.py build/nuscenes-detections /data/nuscenes/pointpillars_10sweep/manifest.json evaluation/nuscenes_submission.json
	python3 tools/evaluate_nuscenes.py evaluation/nuscenes_submission.json --output evaluation/nuscenes-mini

clean:
	rm -rf build
