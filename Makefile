CC ?= cc
PYTHON ?= python3
CFLAGS ?= -O3 -march=native -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS += -Iinclude
CFLAGS += -pthread
LDLIBS += -pthread
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
OMP ?= 0
CPPFLAGS += -DPP_WITH_ACCELERATE
LDLIBS += -framework Accelerate
CPU_EXTRA_SRC := src/infer_apple.c
CXX_RUNTIME := -lc++
GGML_RPATH := -Wl,-rpath,@loader_path/ggml-install/lib
TUI_LIBS :=
TUI_TARGET := build/pointpillars
TUI_BINARY ?= ./build/pointpillars
TUI_MODE ?= tui
else
OMP ?= 1
CPU_EXTRA_SRC :=
CXX_RUNTIME := -lstdc++
GGML_RPATH := -Wl,-rpath,'$$ORIGIN/ggml-install/lib'
TUI_LIBS := -lutil
TUI_TARGET := build/pointpillars_cudnn
TUI_BINARY ?= ./build/pointpillars_cudnn
TUI_MODE ?= tui-cuda
endif
ifeq ($(OMP),1)
CFLAGS += -fopenmp
LDLIBS += -fopenmp
else
CFLAGS += -Wno-unknown-pragmas
endif
MODEL ?= nuscenes_multihead.ppw
CHECKPOINT ?= ckpts/pp_multihead_nds5823_updated.pth
CONFIG ?= cfgs/pointpillars.yaml
NUSCENES_ROOT ?= /data/nuscenes
PREPARED_DATA ?= $(NUSCENES_ROOT)/pointpillars_10sweep
EVAL_PYTHON ?= $(PYTHON)
EVAL_SPLIT ?= mini_val
DETECTIONS_DIR ?= build/nuscenes-detections
SUBMISSION ?= build/nuscenes-submission.json
EVAL_OUTPUT ?= build/nuscenes-evaluation
NVCC ?= /usr/local/cuda-12.4/bin/nvcc
CUDA_ARCH ?= sm_89
GGML_VERSION ?= v0.16.0
GGML_COMMIT ?= 524f974bb21a1013408f76d71c15732482c0c3fe
GGML_SOURCE ?= build/ggml-src
HASH_CMD := $(if $(shell command -v sha256sum 2>/dev/null),sha256sum,shasum -a 256)
GGML_SOURCE_ID := $(shell printf '%s\n' '$(abspath $(GGML_SOURCE))' | $(HASH_CMD) | cut -c1-12)
GGML_BUILD_DIR ?= build/ggml-build/$(GGML_SOURCE_ID)
GGML_INSTALL ?= build/ggml-install
PERF_FRAME ?= $(shell find $(PREPARED_DATA) -name '*.bin' -type f 2>/dev/null | sort | head -1)
PERF_REPS ?= 10
PERF_THREADS ?= 16
TUI_DATA ?= $(PREPARED_DATA)
CPU_CONFIG_ID := $(shell printf '%s\n' '$(CC)|$(CPPFLAGS)|$(CFLAGS)|$(LDLIBS)|OMP=$(OMP)' | $(HASH_CMD) | cut -c1-12)
CUDA_CONFIG_ID := $(shell printf '%s\n' '$(NVCC)|$(CPPFLAGS)|$(CFLAGS)|$(CUDA_ARCH)|OMP=$(OMP)' | $(HASH_CMD) | cut -c1-12)
CUDNN_CONFIG_ID := $(shell printf '%s\n' '$(NVCC)|$(CPPFLAGS)|$(CFLAGS)|$(CUDA_ARCH)|OMP=$(OMP)|cuDNN' | $(HASH_CMD) | cut -c1-12)
GGML_CONFIG_ID := $(shell printf '%s\n' '$(CC)|$(CPPFLAGS)|$(CFLAGS)|$(LDLIBS)|OMP=$(OMP)|$(GGML_COMMIT)' | $(HASH_CMD) | cut -c1-12)
CPU_BINARY := build/pointpillars.$(CPU_CONFIG_ID)
CUDA_BINARY := build/pointpillars_cuda.$(CUDA_CONFIG_ID)
CUDA_OBJDIR := build/cuda/$(CUDA_CONFIG_ID)
CUDNN_BINARY := build/pointpillars_cudnn.$(CUDNN_CONFIG_ID)
CUDNN_OBJDIR := build/cudnn/$(CUDNN_CONFIG_ID)
GGML_BINARY := build/pointpillars_ggml.$(GGML_CONFIG_ID)
GGML_STAMP := $(GGML_INSTALL)/.pointpillars-$(GGML_COMMIT)

.PHONY: all model setup-model cuda cudnn cudnn-test ggml test portable-test perf perf-cpu perf-cuda perf-cuda-compact perf-cudnn perf-cudnn-compact perf-ggml tui-video prepare-data checkpoint-oracle checkpoint-oracle-cuda checkpoint-oracle-cudnn checkpoint-oracle-ggml evaluate evaluate-cpu evaluate-cuda clean FORCE
all: build/pointpillars
cuda: build/pointpillars_cuda
cudnn: build/pointpillars_cudnn
ggml: build/pointpillars_ggml

FORCE:

model: $(MODEL)
setup-model:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -r requirements-export.txt
$(MODEL): tools/export_checkpoint.py $(CHECKPOINT) $(CONFIG)
	$(PYTHON) tools/export_checkpoint.py $(CHECKPOINT) $(CONFIG) $@

$(CPU_BINARY): src/main.c src/model.c src/voxel.c src/infer_cpu.c $(CPU_EXTRA_SRC) src/decode.c src/tui.c include/pp_model.h include/pointpillars.h include/pp_infer.h include/pp_decode.h include/pp_tui.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/main.c src/model.c src/voxel.c src/infer_cpu.c $(CPU_EXTRA_SRC) src/decode.c src/tui.c -lm $(LDLIBS) -o $@

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
	$(CC) $(CPPFLAGS) $(CFLAGS) -DPP_WITH_CUDA -DPP_WITH_CUDNN -c src/main.c -o $(CUDNN_OBJDIR)/main.o
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
	$(CC) $(CPPFLAGS) -I$(GGML_INSTALL)/include $(CFLAGS) -DPP_WITH_GGML src/main.c src/model.c src/voxel.c src/infer_cpu.c $(CPU_EXTRA_SRC) src/infer_ggml.c src/decode.c src/tui.c -L$(GGML_INSTALL)/lib $(GGML_RPATH) -lggml -lggml-cpu -lggml-base $(CXX_RUNTIME) -lm $(LDLIBS) -o $@

build/pointpillars_ggml: $(GGML_BINARY) FORCE
	cp $(GGML_BINARY) $@

build/test_model: src/model.c tests/test_model.c include/pp_model.h FORCE
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/model.c tests/test_model.c -o $@

build/test_voxel: src/voxel.c tests/test_voxel.c include/pointpillars.h FORCE
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/voxel.c tests/test_voxel.c -lm -o $@

build/test_decode: src/decode.c src/infer_cpu.c $(CPU_EXTRA_SRC) src/model.c tests/test_decode.c include/pp_decode.h FORCE
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/decode.c src/infer_cpu.c $(CPU_EXTRA_SRC) src/model.c tests/test_decode.c -lm $(LDLIBS) -o $@

build/test_cpu_conv: src/infer_cpu.c $(CPU_EXTRA_SRC) src/model.c tests/test_cpu_conv.c include/pp_kernels.h FORCE
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/infer_cpu.c $(CPU_EXTRA_SRC) src/model.c tests/test_cpu_conv.c -lm $(LDLIBS) -o $@

build/test_tui: src/tui.c tests/test_tui.c include/pp_tui.h FORCE
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/tui.c tests/test_tui.c -lm $(TUI_LIBS) $(LDLIBS) -o $@

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
	$(PYTHON) tools/perf.py run --backend cpu --binary ./build/pointpillars --model $(MODEL) --points "$(PERF_FRAME)" --reps $(PERF_REPS) --threads $(PERF_THREADS) --output build/perf/cpu.json

perf-cuda: build/pointpillars_cuda
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	mkdir -p build/perf
	$(PYTHON) tools/perf.py run --backend cuda --binary ./build/pointpillars_cuda --model $(MODEL) --points "$(PERF_FRAME)" --reps $(PERF_REPS) --output build/perf/cuda.json

perf-cuda-compact: build/pointpillars_cuda
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	mkdir -p build/perf
	$(PYTHON) tools/perf.py run --backend cuda --output-mode compact --binary ./build/pointpillars_cuda --model $(MODEL) --points "$(PERF_FRAME)" --reps $(PERF_REPS) --output build/perf/cuda-compact.json

perf-cudnn: build/pointpillars_cudnn
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	mkdir -p build/perf
	$(PYTHON) tools/perf.py run --backend cuda --binary ./build/pointpillars_cudnn --model $(MODEL) --points "$(PERF_FRAME)" --reps $(PERF_REPS) --output build/perf/cudnn.json

perf-cudnn-compact: build/pointpillars_cudnn
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	mkdir -p build/perf
	$(PYTHON) tools/perf.py run --backend cuda --output-mode compact --binary ./build/pointpillars_cudnn --model $(MODEL) --points "$(PERF_FRAME)" --reps $(PERF_REPS) --output build/perf/cudnn-compact.json

perf-ggml: build/pointpillars_ggml
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	mkdir -p build/perf
	$(PYTHON) tools/perf.py run --backend cpu --binary ./build/pointpillars_ggml --model $(MODEL) --points "$(PERF_FRAME)" --reps $(PERF_REPS) --threads $(PERF_THREADS) --output build/perf/ggml.json

tui-video: $(TUI_TARGET)
	$(PYTHON) tools/record_tui.py $(TUI_BINARY) $(MODEL) $(TUI_DATA) docs/pointpillars-tui.mp4 --poster docs/pointpillars-tui.png --mode $(TUI_MODE)

prepare-data:
	$(PYTHON) tools/prepare_nuscenes.py --root $(NUSCENES_ROOT) --output $(PREPARED_DATA)

checkpoint-oracle: test
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	OMP_NUM_THREADS=16 ./build/pointpillars infer $(MODEL) "$(PERF_FRAME)" /tmp/nuscenes_cpu.ppout 5
	$(PYTHON) tools/oracle_checkpoint.py $(CHECKPOINT) "$(PERF_FRAME)" /tmp/nuscenes_cpu.ppout

checkpoint-oracle-cuda: test build/pointpillars_cuda
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	PP_CUDA_PRECISE=1 ./build/pointpillars_cuda infer-cuda $(MODEL) "$(PERF_FRAME)" /tmp/nuscenes_cuda.ppout 5
	$(PYTHON) tools/oracle_checkpoint.py $(CHECKPOINT) "$(PERF_FRAME)" /tmp/nuscenes_cuda.ppout

checkpoint-oracle-cudnn: test build/pointpillars_cudnn
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	PP_CUDA_PRECISE=1 ./build/pointpillars_cudnn infer-cuda $(MODEL) "$(PERF_FRAME)" /tmp/nuscenes_cudnn.ppout 5
	$(PYTHON) tools/oracle_checkpoint.py $(CHECKPOINT) "$(PERF_FRAME)" /tmp/nuscenes_cudnn.ppout

checkpoint-oracle-ggml: test build/pointpillars_ggml
	@test -n "$(PERF_FRAME)" || { echo "PERF_FRAME is required (no prepared nuScenes frame found)" >&2; exit 2; }
	OMP_NUM_THREADS=16 ./build/pointpillars_ggml infer $(MODEL) "$(PERF_FRAME)" /tmp/nuscenes_ggml.ppout 5
	$(PYTHON) tools/oracle_checkpoint.py $(CHECKPOINT) "$(PERF_FRAME)" /tmp/nuscenes_ggml.ppout

evaluate: evaluate-cpu

evaluate-cpu: build/pointpillars
	mkdir -p $(DETECTIONS_DIR) $(dir $(SUBMISSION)) $(EVAL_OUTPUT)
	./build/pointpillars batch $(MODEL) $(PREPARED_DATA) $(DETECTIONS_DIR)
	$(EVAL_PYTHON) tools/make_submission.py $(DETECTIONS_DIR) $(PREPARED_DATA)/manifest.json $(SUBMISSION) --root $(NUSCENES_ROOT) --split $(EVAL_SPLIT)
	$(EVAL_PYTHON) tools/evaluate_nuscenes.py $(SUBMISSION) --root $(NUSCENES_ROOT) --output $(EVAL_OUTPUT)

evaluate-cuda: build/pointpillars_cudnn
	mkdir -p $(DETECTIONS_DIR) $(dir $(SUBMISSION)) $(EVAL_OUTPUT)
	./build/pointpillars_cudnn batch-cuda $(MODEL) $(PREPARED_DATA) $(DETECTIONS_DIR)
	$(EVAL_PYTHON) tools/make_submission.py $(DETECTIONS_DIR) $(PREPARED_DATA)/manifest.json $(SUBMISSION) --root $(NUSCENES_ROOT) --split $(EVAL_SPLIT)
	$(EVAL_PYTHON) tools/evaluate_nuscenes.py $(SUBMISSION) --root $(NUSCENES_ROOT) --output $(EVAL_OUTPUT)

clean:
	rm -rf build
