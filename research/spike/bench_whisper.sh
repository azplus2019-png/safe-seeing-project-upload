#!/bin/bash
# Spike tool for benchmarking whisper.cpp performance
git clone https://github.com/ggerganov/whisper.cpp.git --depth 1
cd whisper.cpp
make -j
bash ./models/download-ggml-model.sh base.en
./build/bin/whisper-bench -m models/ggml-base.en.bin -t 2
