#include <iostream>
#include <random>
#include <immintrin.h>
#include <thread>
#include <vector>
#include <chrono>
#include <cstring>  
using namespace std;

bool multithreading = false; 
bool simd = false;
bool cache_optimization = false;
bool everything = false;
int matrix_size = 1000; 
int block_size = 64;   
int thread_num = 1;  
float matrix_sparsity_A = 0.1; 
float matrix_sparsity_B = 0.1; 

mt19937 gen(666); 
uniform_real_distribution<float> dis(0.0, 1.0);

// generate a sparse matrix
void gen_matrix(vector<vector<float>>& matrix, int size, float density) {
    for (int i = 0; i < size; ++i) 
    {
        for (int j = 0; j < size; ++j) 
        {
            matrix[i][j] = (dis(gen) < density) ? dis(gen) : 0.0f;
        }
    }
}

// matrix multiplication
void multiply_matrix(const vector<vector<float>>& A, const vector<vector<float>>& B, vector<vector<float>>& C, int start, int end) 
{
    for (int i = start; i < end; ++i) 
    {
        for (int j = 0; j < matrix_size; ++j) 
        {
            float sum = 0;
            for (int k = 0; k < matrix_size; ++k) 
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// multi-threaded matrix multiplication
void thread_matrix(const vector<vector<float>>& A, const vector<vector<float>>& B, vector<vector<float>>& C)
{
    int rows_per_thread = matrix_size / thread_num;
    vector<thread> threads;
    for (int t = 0; t < thread_num; ++t) 
    {
        int start = t * rows_per_thread;
        int end = (t == thread_num - 1) ? matrix_size : (t + 1) * rows_per_thread;
        threads.emplace_back(multiply_matrix, ref(A), ref(B), ref(C), start, end);
    }
    for (auto& thread : threads) 
    {
        thread.join();
    }
}

// Cache-optimized matrix multiplication
void cache_optimized_matrix(const vector<vector<float>>& A, const vector<vector<float>>& B, vector<vector<float>>& C) {
    for (int i = 0; i < matrix_size; i += block_size) 
    {
        for (int j = 0; j < matrix_size; j += block_size) 
        {
            for (int k = 0; k < matrix_size; k += block_size) 
            {
                for (int x = i; x < min(i + block_size, matrix_size); ++x) 
                {
                    for (int y = j; y < min(j + block_size, matrix_size); ++y) 
                    {
                        float sum = 0;
                        for (int z = k; z < min(k + block_size, matrix_size); ++z) 
                        {
                            sum += A[x][z] * B[z][y];
                        }
                        C[x][y] += sum;
                    }
                }
            }
        }
    }
}

// SIMD-enhanced matrix multiplication
void simd_matrix(const vector<vector<float>>& A, const vector<vector<float>>& B, vector<vector<float>>& C, int start, int end) 
{
    for (int i = start; i < end; ++i) 
    {
        for (int j = 0; j < matrix_size; j += 8) 
        { 
            __m256 sum_vector = _mm256_setzero_ps();  

            for (int k = 0; k < matrix_size; ++k) 
            {
                __m256 a_vector = _mm256_broadcast_ss(&A[i][k]);  
                __m256 b_vector = _mm256_loadu_ps(&B[k][j]);       
                sum_vector = _mm256_fmadd_ps(a_vector, b_vector, sum_vector); 
            }
            _mm256_storeu_ps(&C[i][j], sum_vector);
        }
    }
}

// SIMD within a thread to multiply matrix blocks
void simd_thread_blocks(const vector<vector<float>>& A, const vector<vector<float>>& B, vector<vector<float>>& C, int start, int end, int n) {
    for (int i = start; i < end; i += block_size) 
    {
        for (int j = 0; j < n; j += block_size) 
        {
            for (int k = 0; k < n; k += block_size) 
            {
                for (int x = i; x < min(i + block_size, end); ++x) 
                {
                    for (int y = j; y < min(j + block_size, n); y += 8) 
                    {
                        __m256 sum_vector = _mm256_setzero_ps();
                        for (int z = k; z < min(k + block_size, n); ++z) 
                        {
                            __m256 a_vector = _mm256_broadcast_ss(&A[x][z]);
                            __m256 b_vector = _mm256_loadu_ps(&B[z][y]);
                            sum_vector = _mm256_fmadd_ps(a_vector, b_vector, sum_vector);
                        }
                        _mm256_storeu_ps(&C[x][y], sum_vector);
                    }
                }
            }
        }
    }
}

void parallel(const vector<vector<float>>& A, const vector<vector<float>>& B, vector<vector<float>>& C, int thread_num) {
    int size = A.size(); 
    vector<thread> threads;

    int rows_per_thread = size / thread_num;
    for (int t = 0; t < thread_num; ++t) 
    {
        int start = t * rows_per_thread;
        int end = (t == thread_num - 1) ? size : (t + 1) * rows_per_thread;
        threads.push_back(thread(simd_thread_blocks, cref(A), cref(B), ref(C), start, end, size));
    }
    for (auto& thread : threads) 
    {
        if (thread.joinable()) 
        {
            thread.join();
        }
    }
}


void control(const vector<vector<float>>& A, const vector<vector<float>>& B, vector<vector<float>>& C) 
{
    if (multithreading) 
    {
        thread_matrix(A, B, C);
    } 
    else if (cache_optimization) 
    {
        cache_optimized_matrix(A, B, C);
    } 
    else if (simd) 
    {
        simd_matrix(A, B, C, 0, matrix_size);
    } 
    else 
    {
        multiply_matrix(A, B, C, 0, matrix_size);
    }
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) 
        {
            if (strcmp(argv[i], "size") == 0 && i + 1 < argc) 
            {
                matrix_size = stoi(argv[++i]);
            } 
            else if (strcmp(argv[i], "density") == 0 && i + 1 < argc) 
            {
                matrix_sparsity_A = matrix_sparsity_B = stof(argv[++i]);
            } 
            else if (strcmp(argv[i], "density_A") == 0 && i + 1 < argc) 
            {
                matrix_sparsity_A = stof(argv[++i]);
            }
            else if (strcmp(argv[i], "density_B") == 0 && i + 1 < argc) 
            {
                matrix_sparsity_B = stof(argv[++i]);
            }
            else if (strcmp(argv[i], "threads") == 0 && i + 1 < argc) 
            {
                thread_num = stoi(argv[++i]);
                multithreading = true;
            } 
            else if (strcmp(argv[i], "simd") == 0) 
            {
                simd = true;
            } 
            else if (strcmp(argv[i], "cache") == 0) 
            {
                cache_optimization = true;
            }
            else if (strcmp(argv[i], "all") == 0 && i + 1 < argc) 
            {
                thread_num = stoi(argv[++i]);
                everything = true;
            }
        }
    }

    vector<vector<float>> A(matrix_size, vector<float>(matrix_size));
    vector<vector<float>> B(matrix_size, vector<float>(matrix_size));
    vector<vector<float>> C(matrix_size, vector<float>(matrix_size, 0.0f));

    gen_matrix(A, matrix_size, matrix_sparsity_A);
    gen_matrix(B, matrix_size, matrix_sparsity_B);

    auto start = chrono::high_resolution_clock::now();
    if (everything == true)
    {
        parallel(A, B, C, thread_num);
    } 
    else
    {
        control(A, B, C);
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Size: " << matrix_size << "x" << matrix_size <<endl;
    cout << "Density: " << matrix_sparsity_A*100 << "% " << "x " << matrix_sparsity_B*100 << "%" <<endl;
    cout << "Threads: " << thread_num << "\n" << endl;
    cout << "Matrix multiplication completed in " << elapsed.count() << " seconds." << endl;

    return 0;
}