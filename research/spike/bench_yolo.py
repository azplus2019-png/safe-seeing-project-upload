import onnxruntime as ort
import numpy as np
import time
import sys

def benchmark_onnx(model_path, iterations=100):
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Model: {model_path}")
    print(f"Input shape: {input_shape}")

    # Generate dummy input matching the shape (handling dynamic axes if any)
    fixed_shape = [s if isinstance(s, int) else 1 for s in input_shape]
    dummy_input = np.random.random(fixed_shape).astype(np.float32)

    # Warmup
    session.run(None, {input_name: dummy_input})

    start = time.time()
    for _ in range(iterations):
        session.run(None, {input_name: dummy_input})
    end = time.time()

    latency = (end - start) / iterations * 1000
    print(f"Average latency over {iterations} iterations: {latency:.2f}ms")
    print(f"FPS equivalent: {1000/latency:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bench_yolo.py <path_to_onnx_model>")
    else:
        benchmark_onnx(sys.argv[1])
