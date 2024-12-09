import numpy as np
import cupy as cp
import time
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Function to perform CPU matrix multiplication
def cpu_matrix_multiplication(A, B):
    start_time = time.perf_counter()  # High-precision timer
    C_cpu = np.dot(A, B)  # Matrix multiplication on CPU
    end_time = time.perf_counter()
    cpu_time = end_time - start_time
    return cpu_time

# Function to perform GPU matrix multiplication
def gpu_matrix_multiplication(A, B):
    # Transfer data to GPU
    A_gpu = cp.array(A)
    B_gpu = cp.array(B)
    start_time = time.perf_counter()  # High-precision timer
    C_gpu = cp.dot(A_gpu, B_gpu)  # Matrix multiplication on GPU
    cp.cuda.Stream.null.synchronize()  # Wait for GPU computation to finish
    end_time = time.perf_counter()
    gpu_time = end_time - start_time
    return gpu_time

# Main experiment function
def run_experiment(matrix_size):
    print(f"Running experiment for matrix size: {matrix_size}x{matrix_size}")

    # Generate random matrices
    A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    B = np.random.rand(matrix_size, matrix_size).astype(np.float32)

    # CPU computation
    cpu_time = cpu_matrix_multiplication(A, B)
    print(f"CPU Time: {cpu_time:.8f} seconds")

    # GPU computation
    gpu_time = gpu_matrix_multiplication(A, B)
    print(f"GPU Time: {gpu_time:.8f} seconds")

    # Return results
    return cpu_time, gpu_time

# Function to plot results
def plot_results(matrix_sizes, cpu_times, gpu_times):
    plt.figure(figsize=(10, 6))
    plt.plot(matrix_sizes, cpu_times, label="CPU Time", marker='o')
    plt.plot(matrix_sizes, gpu_times, label="GPU Time", marker='o')
    plt.xlabel("Matrix Size (NxN)")
    plt.ylabel("Time (seconds)")
    plt.title("CPU vs GPU Matrix Multiplication Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the experiment for different matrix sizes
if __name__ == "__main__":
    matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]  # List of matrix sizes to test
    cpu_times = []
    gpu_times = []

    for size in matrix_sizes:
        cpu_time, gpu_time = run_experiment(size)
        cpu_times.append(cpu_time)
        gpu_times.append(gpu_time)

    # Plot results
    plot_results(matrix_sizes, cpu_times, gpu_times)

def cpu_dense_matrix_multiplication(A, B):
    start_time = time.time()
    C = A.dot(B)
    end_time = time.time()
    return C, end_time - start_time

def gpu_dense_matrix_multiplication(A, B):
    A_gpu = cp.asarray(A)
    B_gpu = cp.asarray(B)
    start_time = time.time()
    C_gpu = cp.dot(A_gpu, B_gpu)
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    return C_gpu, end_time - start_time

def cpu_sparse_matrix_multiplication(A, B):
    start_time = time.time()
    C = A.dot(B)
    end_time = time.time()
    return C, end_time - start_time

def gpu_sparse_matrix_multiplication(A, B):
    A_gpu = cp.sparse.csr_matrix(A)
    B_gpu = cp.sparse.csr_matrix(B)
    start_time = time.time()
    C_sparse_gpu = A_gpu.dot(B_gpu)
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    return C_sparse_gpu, end_time - start_time

def run_experiment(size, sparsity):
    print(f"Running experiment for matrix size: {size}x{size}, sparsity: {sparsity}")
    
    # Generate random dense matrices
    A_dense = np.random.rand(size, size)
    B_dense = np.random.rand(size, size)

    # Generate random sparse matrices
    A_sparse = sp.random(size, size, density=sparsity, format='csr')
    B_sparse = sp.random(size, size, density=sparsity, format='csr')

    # CPU Dense Matrix Multiplication
    _, cpu_dense_time = cpu_dense_matrix_multiplication(A_dense, B_dense)

    # GPU Dense Matrix Multiplication
    _, gpu_dense_time = gpu_dense_matrix_multiplication(A_dense, B_dense)

    # CPU Sparse Matrix Multiplication
    _, cpu_sparse_time = cpu_sparse_matrix_multiplication(A_sparse, B_sparse)

    # GPU Sparse Matrix Multiplication
    try:
        _, gpu_sparse_time = gpu_sparse_matrix_multiplication(A_sparse, B_sparse)
    except cp.cuda.memory.OutOfMemoryError:
        gpu_sparse_time = None

    print(f"Results for matrix size {size}x{size}, sparsity {sparsity}:")
    print(f"  CPU Dense Time: {cpu_dense_time:.8f} seconds")
    print(f"  GPU Dense Time: {gpu_dense_time:.8f} seconds")
    print(f"  CPU Sparse Time: {cpu_sparse_time:.8f} seconds")
    if gpu_sparse_time is not None:
        print(f"  GPU Sparse Time: {gpu_sparse_time:.8f} seconds")
    else:
        print(f"  GPU Sparse Time: Out of Memory")

    return cpu_dense_time, gpu_dense_time, cpu_sparse_time, gpu_sparse_time

def main():
    sizes = [32, 64, 128, 256, 512, 1024]
    sparsities = [1.0, 0.1, 0.01]

    results = []
    for sparsity in sparsities:
        print(f"\nRunning experiments for sparsity level: {sparsity}\n")
        for size in sizes:
            results.append((size, sparsity, *run_experiment(size, sparsity)))

    # Plotting the results
    for sparsity in sparsities:
        filtered_results = [r for r in results if r[1] == sparsity]
        sizes = [r[0] for r in filtered_results]
        cpu_dense_times = [r[2] for r in filtered_results]
        gpu_dense_times = [r[3] for r in filtered_results]
        cpu_sparse_times = [r[4] for r in filtered_results]
        gpu_sparse_times = [r[5] if r[5] is not None else float('nan') for r in filtered_results]

        plt.figure()
        plt.plot(sizes, cpu_dense_times, 'o-', label='CPU Dense Time')
        plt.plot(sizes, gpu_dense_times, 'o-', label='GPU Dense Time')
        plt.plot(sizes, cpu_sparse_times, 'o-', label='CPU Sparse Time')
        plt.plot(sizes, gpu_sparse_times, 'o-', label='GPU Sparse Time')
        plt.xlabel('Matrix Size (NxN)')
        plt.ylabel('Time (seconds)')
        plt.title(f'CPU vs GPU Performance for Matrix Multiplication (Sparsity: {sparsity})')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()

def run_batch_multiplications(matrix_size, num_batches):
    """Run multiple matrix multiplications in a batch and time them."""
    # Generate random matrices
    A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    B = np.random.rand(matrix_size, matrix_size).astype(np.float32)

    # CPU batch computation
    cpu_start_time = time.perf_counter()
    for _ in range(num_batches):
        cpu_matrix_multiplication(A, B)
    cpu_end_time = time.perf_counter()
    cpu_total_time = cpu_end_time - cpu_start_time
    cpu_avg_time = cpu_total_time / num_batches

    # GPU batch computation
    gpu_start_time = time.perf_counter()
    for _ in range(num_batches):
        gpu_matrix_multiplication(A, B)
    gpu_end_time = time.perf_counter()
    gpu_total_time = gpu_end_time - gpu_start_time
    gpu_avg_time = gpu_total_time / num_batches

    return cpu_avg_time, gpu_avg_time

def plot_batch_results(batch_sizes, cpu_avg_times, gpu_avg_times):
    """Plot the average computation time for CPU and GPU for different batch sizes."""
    plt.figure(figsize=(12, 8))
    plt.plot(batch_sizes, cpu_avg_times, label="CPU Avg Time", marker='o')
    plt.plot(batch_sizes, gpu_avg_times, label="GPU Avg Time", marker='o')
    plt.xlabel("Batch Size")
    plt.ylabel("Average Time per Multiplication (seconds)")
    plt.title("Average Time per Multiplication for Different Batch Sizes")
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def extended_experiment():
    matrix_sizes = [128, 256, 512, 1024, 2048,]  # Different matrix sizes to test
    batch_sizes = [1, 10, 50, 100, 500, 1000, 2000, 5000]  # Extended batch sizes

    for matrix_size in matrix_sizes:
        cpu_avg_times = []
        gpu_avg_times = []
        print(f"\nRunning experiment for matrix size {matrix_size}x{matrix_size}...")

        for batch_size in batch_sizes:
            print(f"  Running batch of {batch_size} multiplications...")
            cpu_avg_time, gpu_avg_time = run_batch_multiplications(matrix_size, batch_size)
            print(f"    CPU Avg Time: {cpu_avg_time:.8f} seconds, GPU Avg Time: {gpu_avg_time:.8f} seconds")
            cpu_avg_times.append(cpu_avg_time)
            gpu_avg_times.append(gpu_avg_time)

        plt.figure(figsize=(12, 8))
        plt.plot(batch_sizes, cpu_avg_times, label="CPU Avg Time", marker='o')
        plt.plot(batch_sizes, gpu_avg_times, label="GPU Avg Time", marker='o')
        plt.xlabel("Batch Size")
        plt.ylabel("Average Time per Multiplication (seconds)")
        plt.title(f"Performance for Matrix Size {matrix_size}x{matrix_size}")
        plt.xscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    extended_experiment()

if __name__ == "__main__":
    main()
