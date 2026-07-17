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
GGML_VERSION ?= v0.16.0
GGML_COMMIT ?= 524f974bb21a1013408f76d71c15732482c0c3fe
GGML_SOURCE ?= build/ggml-src
GGML_SOURCE_ID := $(shell printf '%s\n' '$(abspath $(GGML_SOURCE))' | sha256sum | cut -c1-12)
GGML_BUILD_DIR ?= build/ggml-build/$(GGML_SOURCE_ID)
GGML_INSTALL ?= build/ggml-install
PERF_FRAME ?= $(shell find /data/nuscenes/pointpillars_10sweep -name '*.bin' -type f 2>/dev/null | sort | head -1)
PERF_REPS ?= 10
PERF_THREADS ?= 16
CPU_CONFIG_ID := $(shell printf '%s\n' '$(CC)|$(CPPFLAGS)|$(CFLAGS)|$(LDLIBS)|OMP=$(OMP)' | sha256sum | cut -c1-12)
CUDA_CONFIG_ID := $(shell printf '%s\n' '$(NVCC)|$(CPPFLAGS)|$(CFLAGS)|$(CUDA_ARCH)|OMP=$(OMP)' | sha256sum | cut -c1-12)
CUDNN_CONFIG_ID := $(shell printf '%s\n' '$(NVCC)|$(CPPFLAGS)|$(CFLAGS)|$(CUDA_ARCH)|OMP=$(OMP)|cuDNN' | sha256sum | cut -c1-12)
GGML_CONFIG_ID := $(shell printf '%s\n' '$(CC)|$(CPPFLAGS)|$(CFLAGS)|$(LDLIBS)|OMP=$(OMP)|$(GGML_COMMIT)' | sha256sum | cut -c1-12)
CPU_BINARY := build/pointpillars.$(CPU_CONFIG_ID)
CUDA_BINARY := build/pointpillars_cuda.$(CUDA_CONFIG_ID)
CUDA_OBJDIR := build/cuda/$(CUDA_CONFIG_ID)
CUDNN_BINARY := build/pointpillars_cudnn.$(CUDNN_CONFIG_ID)
CUDNN_OBJDIR := build/cudnn/$(CUDNN_CONFIG_ID)
GGML_BINARY := build/pointpillars_ggml.$(GGML_CONFIG_ID)
GGML_STAMP := $(GGML_INSTALL)/.pointpillars-$(GGML_COMMIT)

.PHONY: all model cuda cudnn cudnn-test ggml test portable-test perf perf-cpu perf-cuda perf-cuda-compact perf-cudnn perf-cudnn-compact perf-ggml prepare-data checkpoint-oracle checkpoint-oracle-cuda checkpoint-oracle-cudnn checkpoint-oracle-ggml evaluate clean FORCE
all: build/pointpillars
cuda: build/pointpillars_cuda
cudnn: build/pointpillars_cudnn
ggml: build/pointpillars_ggml

FORCE:

model: $(MODEL)
$(MODEL): tools/export_checkpoint.py $(CHECKPOINT) $(CONFIG)
	python3 tools/export_checkpoint.py $(CHECKPOINT) $(CONFIG) $@

$(CPU_BINARY): src/main.c src/model.c src/voxel.c src/infer_cpu.c src/decode.c src/tui.c include/pp_model.h include/pointpillars.h include/pp_infer.h include/pp_decode.h include/pp_tui.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/main.c src/model.c src/voxel.c src/infer_cpu.c src/decode.c src/tui.c -lm $(LDLIBS) -o $@

# The public path is refreshed from the active configuration-specific binary.
# This avoids relying on timestamp ordering when switching OMP or compiler flags.
build/pointpillars: $(CPU_BINARY) FORCE
	cp $(CPU_BINARY) $@

$(CUDA_BINARY): src/main.c src/model.c src/voxel.c src/infer_cpu.c src/decode.c src/tui.c src/infer_cuda.cu include/pp_cuda.h
	mkdir -p $(CUDA_OBJDIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DPP_WITH_CUDA -c src/main.c -o $(CUDA_OBJDIR)/main.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/model.c -o $(CUDA_OBJDIR)/model.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/voxel.c -o $(CUDA_OBJDIR)/voxel.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/infer_cpu.c -o $(CUDA_OBJDIR)/infer_cpu.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/decode.c -o $(CUDA_OBJDIR)/decode.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/tui.c -o $(CUDA_OBJDIR)/tui.o
	$(NVCC) $(CPPFLAGS) -O3 -arch=$(CUDA_ARCH) -c src/infer_cuda.cu -o $(CUDA_OBJDIR)/infer_cuda.o
	$(NVCC) -Xcompiler -fopenmp -Xcompiler -pthread $(CUDA_OBJDIR)/*.o -lgomp -lm -o $@

build/pointpillars_cuda: $(CUDA_BINARY) FORCE
	cp $(CUDA_BINARY) $@

$(CUDNN_BINARY): src/main.c src/model.c src/voxel.c src/infer_cpu.c src/decode.c src/tui.c src/infer_cuda.cu src/infer_cudnn.cu include/pp_cuda.h include/pp_cudnn.h
	mkdir -p $(CUDNN_OBJDIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DPP_WITH_CUDA -c src/main.c -o $(CUDNN_OBJDIR)/main.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/model.c -o $(CUDNN_OBJDIR)/model.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/voxel.c -o $(CUDNN_OBJDIR)/voxel.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/infer_cpu.c -o $(CUDNN_OBJDIR)/infer_cpu.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/decode.c -o $(CUDNN_OBJDIR)/decode.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/tui.c -o $(CUDNN_OBJDIR)/tui.o
	$(NVCC) $(CPPFLAGS) -O3 -arch=$(CUDA_ARCH) -DPP_WITH_CUDNN -c src/infer_cuda.cu -o $(CUDNN_OBJDIR)/infer_cuda.o
	$(NVCC) $(CPPFLAGS) -O3 -arch=$(CUDA_ARCH) -c src/infer_cudnn.cu -o $(CUDNN_OBJDIR)/infer_cudnn.o
	$(NVCC) -Xcompiler -fopenmp -Xcompiler -pthread $(CUDNN_OBJDIR)/*.o -lcudnn -lgomp -lm -o $@

build/pointpillars_cudnn: $(CUDNN_BINARY) FORCE
	cp $(CUDNN_BINARY) $@

build/test_cudnn: src/infer_cudnn.cu tests/test_cudnn.cu include/pp_cudnn.h include/pp_model.h FORCE
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=$(CUDA_ARCH) src/infer_cudnn.cu tests/test_cudnn.cu -lcudnn -o $@

cudnn-test: build/test_cudnn
	./build/test_cudnn

$(GGML_SOURCE)/.git:
	mkdir -p $(dir $(GGML_SOURCE))
	git clone --depth 1 --branch $(GGML_VERSION) https://github.com/ggml-org/ggml.git $(GGML_SOURCE)
	git -C $(GGML_SOURCE) checkout $(GGML_COMMIT)

$(GGML_STAMP): $(GGML_SOURCE)/.git
	cmake -S $(GGML_SOURCE) -B $(GGML_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(abspath $(GGML_INSTALL)) -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
	cmake --build $(GGML_BUILD_DIR) --target install -j 8
	mkdir -p $(GGML_INSTALL)
	touch $@

$(GGML_BINARY): $(GGML_STAMP) src/main.c src/model.c src/voxel.c src/infer_cpu.c src/infer_ggml.c src/decode.c src/tui.c include/pp_ggml.h
	mkdir -p build
	$(CC) $(CPPFLAGS) -I$(GGML_INSTALL)/include $(CFLAGS) -DPP_WITH_GGML src/main.c src/model.c src/voxel.c src/infer_cpu.c src/infer_ggml.c src/decode.c src/tui.c -L$(GGML_INSTALL)/lib -Wl,-rpath,'$$ORIGIN/ggml-install/lib' -lggml -lggml-cpu -lggml-base -lstdc++ -lm $(LDLIBS) -o $@

build/pointpillars_ggml: $(GGML_BINARY) FORCE
	cp $(GGML_BINARY) $@

build/test_model: src/model.c tests/test_model.c include/pp_model.h FORCE
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/model.c tests/test_model.c -o $@

build/test_voxel: src/voxel.c tests/test_voxel.c include/pointpillars.h FORCE
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/voxel.c tests/test_voxel.c -lm -o $@

build/test_decode: src/decode.c src/infer_cpu.c src/model.c tests/test_decode.c include/pp_decode.h FORCE
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/decode.c src/infer_cpu.c src/model.c tests/test_decode.c -lm $(LDLIBS) -o $@

build/test_cpu_conv: src/infer_cpu.c src/model.c tests/test_cpu_conv.c include/pp_kernels.h FORCE
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/infer_cpu.c src/model.c tests/test_cpu_conv.c -lm $(LDLIBS) -o $@

build/test_tui: src/tui.c tests/test_tui.c include/pp_tui.h FORCE
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/tui.c tests/test_tui.c -lm $(LDLIBS) -o $@

test: model build/pointpillars build/test_model build/test_voxel build/test_decode build/test_cpu_conv build/test_tui
	./build/test_model $(MODEL)
	./build/test_voxel
	./build/test_decode
	./build/test_cpu_conv
	./build/test_tui

portable-test:
	$(MAKE) clean
	$(MAKE) OMP=0 test

perf: perf-cpu

perf-cpu: build/pointpillars
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	mkdir -p build/perf
	python3 tools/perf.py run --backend cpu --binary ./build/pointpillars --model $(MODEL) --points "$(PERF_FRAME)" --reps $(PERF_REPS) --threads $(PERF_THREADS) --output build/perf/cpu.json

perf-cuda: build/pointpillars_cuda
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	mkdir -p build/perf
	python3 tools/perf.py run --backend cuda --binary ./build/pointpillars_cuda --model $(MODEL) --points "$(PERF_FRAME)" --reps $(PERF_REPS) --output build/perf/cuda.json

perf-cuda-compact: build/pointpillars_cuda
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	mkdir -p build/perf
	python3 tools/perf.py run --backend cuda --output-mode compact --binary ./build/pointpillars_cuda --model $(MODEL) --points "$(PERF_FRAME)" --reps $(PERF_REPS) --output build/perf/cuda-compact.json

perf-cudnn: build/pointpillars_cudnn
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	mkdir -p build/perf
	python3 tools/perf.py run --backend cuda --binary ./build/pointpillars_cudnn --model $(MODEL) --points "$(PERF_FRAME)" --reps $(PERF_REPS) --output build/perf/cudnn.json

perf-cudnn-compact: build/pointpillars_cudnn
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	mkdir -p build/perf
	python3 tools/perf.py run --backend cuda --output-mode compact --binary ./build/pointpillars_cudnn --model $(MODEL) --points "$(PERF_FRAME)" --reps $(PERF_REPS) --output build/perf/cudnn-compact.json

perf-ggml: build/pointpillars_ggml
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	mkdir -p build/perf
	python3 tools/perf.py run --backend cpu --binary ./build/pointpillars_ggml --model $(MODEL) --points "$(PERF_FRAME)" --reps $(PERF_REPS) --threads $(PERF_THREADS) --output build/perf/ggml.json

prepare-data:
	python3 tools/prepare_nuscenes.py --root /data/nuscenes --output /data/nuscenes/pointpillars_10sweep

checkpoint-oracle: test
	@f=$$(find /data/nuscenes/pointpillars_10sweep -name '*.bin' | sort | head -1); \
	OMP_NUM_THREADS=16 ./build/pointpillars infer $(MODEL) "$$f" /tmp/nuscenes_cpu.ppout 5; \
	python3 tools/oracle_checkpoint.py $(CHECKPOINT) "$$f" /tmp/nuscenes_cpu.ppout

checkpoint-oracle-cuda: test build/pointpillars_cuda
	@f=$$(find /data/nuscenes/pointpillars_10sweep -name '*.bin' | sort | head -1); \
	PP_CUDA_PRECISE=1 ./build/pointpillars_cuda infer-cuda $(MODEL) "$$f" /tmp/nuscenes_cuda.ppout 5; \
	python3 tools/oracle_checkpoint.py $(CHECKPOINT) "$$f" /tmp/nuscenes_cuda.ppout

checkpoint-oracle-cudnn: test build/pointpillars_cudnn
	@f=$$(find /data/nuscenes/pointpillars_10sweep -name '*.bin' | sort | head -1); \
	PP_CUDA_PRECISE=1 ./build/pointpillars_cudnn infer-cuda $(MODEL) "$$f" /tmp/nuscenes_cudnn.ppout 5; \
	python3 tools/oracle_checkpoint.py $(CHECKPOINT) "$$f" /tmp/nuscenes_cudnn.ppout

checkpoint-oracle-ggml: test build/pointpillars_ggml
	@f=$$(find /data/nuscenes/pointpillars_10sweep -name '*.bin' | sort | head -1); \
	OMP_NUM_THREADS=16 ./build/pointpillars_ggml infer $(MODEL) "$$f" /tmp/nuscenes_ggml.ppout 5; \
	python3 tools/oracle_checkpoint.py $(CHECKPOINT) "$$f" /tmp/nuscenes_ggml.ppout

evaluate: cuda
	rm -rf build/nuscenes-detections
	./build/pointpillars_cuda batch-cuda $(MODEL) /data/nuscenes/pointpillars_10sweep build/nuscenes-detections
	mkdir -p evaluation
	python3 tools/make_submission.py build/nuscenes-detections /data/nuscenes/pointpillars_10sweep/manifest.json evaluation/nuscenes_submission.json
	python3 tools/evaluate_nuscenes.py evaluation/nuscenes_submission.json --output evaluation/nuscenes-mini

clean:
	rm -rf build
