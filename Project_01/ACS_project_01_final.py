import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

# part 1

def measure_latency(array_size, access_type="read"):
    x_array = np.ones(array_size, dtype=np.float64)
    start_time = time.perf_counter()
    
    if access_type == "read":
        for i in range(len(x_array)):
            j = x_array[i]
    elif access_type == "write":
        for i in range(len(x_array)):
            x_array[i] = i

    end_time = time.perf_counter()
    latency = (end_time - start_time) / len(x_array)  # average latency
    return latency

#part 2

def measure_bandwidth(array_size, access_granularity, read_write_ratio=(1, 0)):
    x_array = np.ones(array_size, dtype=np.float64)
    total_reads = read_write_ratio[0]
    total_writes = read_write_ratio[1]
    
    start_time = time.perf_counter()

    for i in range(0, len(x_array), access_granularity):
        # Read according to the ratio
        for j in range(total_reads):
            j = x_array[i:i + access_granularity]
        
        # Write according to the ratio
        for j in range(total_writes):
            x_array[i:i + access_granularity] = i

    end_time = time.perf_counter()
    total_time = end_time - start_time
    data = array_size * 8  
    bandwidth = data / total_time / (1024**3)  # GigaBytes/sec
    return bandwidth

# helper function for ThreadPoolExecutor
def read_write(arr, start, end, access_type="read"):
    if access_type == "read":
        for i in range(start, end):
            j = arr[i]
    elif access_type == "write":
        for i in range(start, end):
            arr[i] = i

# part 3

def queuing_theory_test(array_size, num_threads, access_type="read"):
    x_array = np.ones(array_size, dtype=np.float64)
    chunk_size = array_size // num_threads
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size
            futures.append(executor.submit(read_write, x_array, start_index, end_index, access_type))

        for f in futures:
            f.result()

    end_time = time.perf_counter()
    latency = (end_time - start_time) / num_threads 
    return latency

# part 4

def multiply_matrix(size):
    x = np.random.rand(size, size)
    y = np.random.rand(size, size)
    
    start_time = time.perf_counter()
    result = np.dot(x, y)
    end_time = time.perf_counter()
    
    exec_time = end_time - start_time
    return exec_time

# part 5

def tlb_miss_test(array_size, stride):
    x_array = np.ones(array_size, dtype=np.float64)
    start_time = time.perf_counter()

    for i in range(0, array_size, stride):
        x_array[i] = x_array[i] * 2

    end_time = time.perf_counter()
    exec_time = end_time - start_time
    return exec_time

# part 1
cache_size = 1024 * 1024 // 8  # fit inside cache
main_memory_size = 100 * cache_size  # spill into main memory

cache_latency_read = measure_latency(cache_size, "read")
cache_latency_write = measure_latency(cache_size, "write")
memory_latency_read = measure_latency(main_memory_size, "read")
memory_latency_write = measure_latency(main_memory_size, "write")

print("1. Comparing the read/write latency of cache and main memory when the queue length is zero\n")
print(f"Cache Read Latency: {cache_latency_read * 1e9} ns")
print(f"Cache Write Latency: {cache_latency_write * 1e9} ns")
print(f"Main Memory Read Latency: {memory_latency_read * 1e9} ns")
print(f"Main Memory Write Latency: {memory_latency_write * 1e9} ns\n")

# part 2
array_size = 10**7
granularities = [64 // 8, 256 // 8, 1024 // 8]  # 64B, 256B, 1024B
ratios = [(1, 0), (0, 1), (7, 3), (5, 5)]  # read-only, write-only, 70:30, 50:50

print("2. Max bandwidth of the main memory under different data access granularity and different read vs. write intensity ratios\n")

for g in granularities:
    for r in ratios:
        bandwidth = measure_bandwidth(array_size, g, r)
        print(f"Granularity: {g * 8}B, Read {r[0]}, Write {r[1]}, Bandwidth: {bandwidth} GB/s")

print("")

# part 3
threads = [1, 2, 4, 8, 16]  # different num of threads
array_size = 10**7

print("3. Experimenting the trade-off between read/write latency and throughput of the main memory\n")

for t in threads:
    latency = queuing_theory_test(array_size, t, "read")
    print(f"Threads: {t}, Latency: {latency * 1e6} µs")

print("")

# part 4
matrix_sizes = [64, 256, 512, 1024, 2048]  # different matrix sizes

print("4. Experimenting the impact of cache miss ratio on the software speed performance\n")

for size in matrix_sizes:
    exec_time = multiply_matrix(size)
    print(f"Matrix Size: {size}, Execution Time: {exec_time} seconds")

print("")

# part 5
array_size = 10**7
strides = [1, 16, 64, 256, 1024]  # different num of strides

print("5. Experimenting the impact of TLB table miss ratio on the software speed performance\n")

for s in strides:
    exec_time = tlb_miss_test(array_size, s)
    print(f"Stride: {s}, Execution Time: {exec_time} seconds")