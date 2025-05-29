#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip> // For std::fixed and std::setprecision
#include <cmath>   // For fabs, needed for verification
#include <fstream> // Added for file output

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Error checking macro for CUDA calls
#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__       \
                      << ": " << cudaGetErrorString(err_) << std::endl;        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// --- Constants ---
const int NUM_RUNS = 20;      // Number of times to run each multiplication for averaging [cite: 12]
const float EPSILON = 1e-4f;  // Epsilon for result verification, adjust if necessary [cite: 12]
const int TILE_DIM = 16;      // Tile dimension for tiled kernel. Experiment with this for your report [cite: 25]

// --- Forward Declarations ---
void initializeMatrix(float* matrix, int N, unsigned int seed);
void serialMatrixMultiply(const float* A, const float* B, float* C, int N);
bool verifyResults(const float* ref_C, const float* C, int N_rows, int N_cols, const std::string& matrix_name);
void gpuWarmup(std::ostream& out); // Modified to take ostream
void printDeviceProperties(std::ostream& out); // Modified to take ostream

// --- Naive CUDA Kernel --- [cite: 4]
// Each thread computes one element of the result matrix C [cite: 5]
__global__ void naiveMatMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// --- Tiled CUDA Kernel (Shared Memory) --- [cite: 6]
// Uses dynamically allocated shared memory.
// Handles arbitrary matrix sizes (N not necessarily a multiple of TILE_DIM) [cite: 7]
// Includes boundary checks [cite: 8]
__global__ void tiledMatMulKernel(const float* A, const float* B, float* C, int N) {
    extern __shared__ float s_data[]; // Dynamic shared memory [cite: 6]
    float* s_A = s_data; // Shared memory for tile of A
    float* s_B = &s_data[TILE_DIM * TILE_DIM]; // Shared memory for tile of B

    int tx = threadIdx.x; // Thread's x index within the block (0 to TILE_DIM-1)
    int ty = threadIdx.y; // Thread's y index within the block (0 to TILE_DIM-1)

    // Global row and column for the element this thread will compute
    int global_row = blockIdx.y * TILE_DIM + ty;
    int global_col = blockIdx.x * TILE_DIM + tx;

    float sum = 0.0f;

    // Loop over the tiles of A and B required to compute the C tile
    for (int tile_idx = 0; tile_idx < (N + TILE_DIM - 1) / TILE_DIM; ++tile_idx) {
        // Load tile of A into shared memory
        // s_A[ty][tx]
        int a_row = blockIdx.y * TILE_DIM + ty;
        int a_col = tile_idx * TILE_DIM + tx;
        if (a_row < N && a_col < N) {
            s_A[ty * TILE_DIM + tx] = A[a_row * N + a_col];
        } else {
            s_A[ty * TILE_DIM + tx] = 0.0f; // Boundary condition [cite: 8]
        }

        // Load tile of B into shared memory
        // s_B[ty][tx]
        int b_row = tile_idx * TILE_DIM + ty;
        int b_col = blockIdx.x * TILE_DIM + tx;
        if (b_row < N && b_col < N) {
            s_B[ty * TILE_DIM + tx] = B[b_row * N + b_col];
        } else {
            s_B[ty * TILE_DIM + tx] = 0.0f; // Boundary condition [cite: 8]
        }

        __syncthreads(); // Synchronize threads in the block after loading tiles

        // Multiply tiles from shared memory
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[ty * TILE_DIM + k] * s_B[k * TILE_DIM + tx];
        }
        __syncthreads(); // Synchronize threads before loading next pair of tiles
    }

    // Write the result to global memory C
    if (global_row < N && global_col < N) {
        C[global_row * N + global_col] = sum;
    }
}


// --- Main Function ---
int main() {
    std::ofstream outFile;
    outFile.open("results_.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open results.txt for writing." << std::endl;
        return EXIT_FAILURE;
    }

    printDeviceProperties(outFile); // Pass outFile to print to file [cite: 17]
    gpuWarmup(outFile); // Pass outFile to print to file [cite: 12]

    outFile << std::fixed << std::setprecision(3);
    outFile << "Starting Matrix Multiplication Performance Comparison..." << std::endl;
    outFile << "TILE_DIM for Tiled Kernel: " << TILE_DIM << std::endl;
    outFile << "Number of runs for averaging: " << NUM_RUNS << std::endl << std::endl;

    outFile << std::left << std::setw(12) << "Matrix Dim"
              << std::setw(15) << "CPU Time(ms)"
              << std::setw(18) << "Naive GPU(ms)"
              << std::setw(15) << "S/N Speedup"
              << std::setw(18) << "Tiled GPU(ms)"
              << std::setw(15) << "S/T Speedup"
              << std::setw(15) << "N/T Speedup"
              << std::setw(15) << "Naive Verify"
              << std::setw(15) << "Tiled Verify"
              << std::endl;
    outFile << std::string(135, '-') << std::endl;

    // Matrix sizes to test [cite: 9]
    std::vector<int> matrix_sizes = {64, 128, 256, 512, 1024}; // Example sizes, ensure N >= 4x4

    unsigned int seed = 12345; // For reproducible matrix initialization

    for (int N : matrix_sizes) {
        long matrix_elements = static_cast<long>(N) * N;
        if (matrix_elements == 0) { 
            std::cerr << "Matrix size N must be greater than 0." << std::endl; // Errors still to cerr
            continue;
        }
        size_t matrix_size_bytes = matrix_elements * sizeof(float);

        // Host memory allocation
        float* h_A = new float[matrix_elements];
        float* h_B = new float[matrix_elements];
        float* h_C_cpu = new float[matrix_elements];      // Result from CPU
        float* h_C_gpu_naive = new float[matrix_elements]; // Result from Naive GPU
        float* h_C_gpu_tiled = new float[matrix_elements]; // Result from Tiled GPU

        initializeMatrix(h_A, N, seed++);
        initializeMatrix(h_B, N, seed++);

        // Device memory allocation
        float *d_A, *d_B, *d_C_naive, *d_C_tiled;
        CUDA_CHECK(cudaMalloc(&d_A, matrix_size_bytes));
        CUDA_CHECK(cudaMalloc(&d_B, matrix_size_bytes));
        CUDA_CHECK(cudaMalloc(&d_C_naive, matrix_size_bytes));
        CUDA_CHECK(cudaMalloc(&d_C_tiled, matrix_size_bytes));

        // Copy A and B to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A, matrix_size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, matrix_size_bytes, cudaMemcpyHostToDevice));

        double avg_cpu_time_ms = 0.0;
        double avg_naive_gpu_time_ms = 0.0;
        double avg_tiled_gpu_time_ms = 0.0;

        // 1. Serial CPU Implementation [cite: 3]
        for (int run = 0; run < NUM_RUNS; ++run) {
            auto start_cpu = std::chrono::high_resolution_clock::now();
            serialMatrixMultiply(h_A, h_B, h_C_cpu, N);
            auto end_cpu = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> cpu_time_ms = end_cpu - start_cpu;
            avg_cpu_time_ms += cpu_time_ms.count();
        }
        avg_cpu_time_ms /= NUM_RUNS;

        // CUDA events for timing GPU kernels [cite: 12]
        cudaEvent_t start_event, stop_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        float elapsed_ms;

        // 2. Naive CUDA Kernel [cite: 4]
        dim3 threadsPerBlockNaive(TILE_DIM, TILE_DIM); 
        dim3 numBlocksNaive((N + threadsPerBlockNaive.x - 1) / threadsPerBlockNaive.x,
                            (N + threadsPerBlockNaive.y - 1) / threadsPerBlockNaive.y);
        
        for (int run = 0; run < NUM_RUNS; ++run) {
            CUDA_CHECK(cudaEventRecord(start_event));
            naiveMatMulKernel<<<numBlocksNaive, threadsPerBlockNaive>>>(d_A, d_B, d_C_naive, N);
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
            avg_naive_gpu_time_ms += elapsed_ms;
        }
        avg_naive_gpu_time_ms /= NUM_RUNS;
        CUDA_CHECK(cudaMemcpy(h_C_gpu_naive, d_C_naive, matrix_size_bytes, cudaMemcpyDeviceToHost));

        // 3. Tiled CUDA Kernel [cite: 6]
        dim3 threadsPerBlockTiled(TILE_DIM, TILE_DIM);
        dim3 numBlocksTiled((N + TILE_DIM - 1) / TILE_DIM,
                            (N + TILE_DIM - 1) / TILE_DIM);
        unsigned int shared_mem_size_bytes = 2 * TILE_DIM * TILE_DIM * sizeof(float); // [cite: 6]

        for (int run = 0; run < NUM_RUNS; ++run) {
            CUDA_CHECK(cudaEventRecord(start_event));
            tiledMatMulKernel<<<numBlocksTiled, threadsPerBlockTiled, shared_mem_size_bytes>>>(d_A, d_B, d_C_tiled, N);
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
            avg_tiled_gpu_time_ms += elapsed_ms;
        }
        avg_tiled_gpu_time_ms /= NUM_RUNS;
        CUDA_CHECK(cudaMemcpy(h_C_gpu_tiled, d_C_tiled, matrix_size_bytes, cudaMemcpyDeviceToHost));

        // Verification [cite: 12]
        bool naive_verified = verifyResults(h_C_cpu, h_C_gpu_naive, N, N, "Naive GPU");
        bool tiled_verified = verifyResults(h_C_cpu, h_C_gpu_tiled, N, N, "Tiled GPU");

        // Speedups [cite: 23]
        double speedup_serial_naive = (avg_naive_gpu_time_ms > 0) ? avg_cpu_time_ms / avg_naive_gpu_time_ms : 0.0;
        double speedup_serial_tiled = (avg_tiled_gpu_time_ms > 0) ? avg_cpu_time_ms / avg_tiled_gpu_time_ms : 0.0;
        double speedup_naive_tiled = (avg_tiled_gpu_time_ms > 0) ? avg_naive_gpu_time_ms / avg_tiled_gpu_time_ms : 0.0;
        
        std::string n_str = std::to_string(N) + "x" + std::to_string(N);
        outFile << std::left << std::setw(12) << n_str
                  << std::setw(15) << avg_cpu_time_ms
                  << std::setw(18) << avg_naive_gpu_time_ms
                  << std::setw(15) << speedup_serial_naive
                  << std::setw(18) << avg_tiled_gpu_time_ms
                  << std::setw(15) << speedup_serial_tiled
                  << std::setw(15) << speedup_naive_tiled
                  << std::setw(15) << (naive_verified ? "OK" : "FAIL")
                  << std::setw(15) << (tiled_verified ? "OK" : "FAIL")
                  << std::endl;

        // Cleanup
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C_naive));
        CUDA_CHECK(cudaFree(d_C_tiled));

        delete[] h_A;
        delete[] h_B;
        delete[] h_C_cpu;
        delete[] h_C_gpu_naive;
        delete[] h_C_gpu_tiled;
    }
    outFile << std::string(135, '-') << std::endl;
    outFile << "All experiments complete." << std::endl;

    outFile.close(); // Close the file
    std::cout << "Results saved to results.txt" << std::endl; // Notify user on console

    return 0;
}

// --- Helper Function Implementations ---

// Initializes a square matrix with random float values. Uses float type for all matrix values [cite: 12]
void initializeMatrix(float* matrix, int N, unsigned int seed) {
    std::mt19937 rng(seed); 
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (long i = 0; i < static_cast<long>(N) * N; ++i) {
        matrix[i] = dist(rng);
    }
}

// Serial matrix multiplication on CPU [cite: 3]
void serialMatrixMultiply(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Verifies results by comparing two matrices element-wise [cite: 12]
bool verifyResults(const float* ref_C, const float* C_to_verify, int N_rows, int N_cols, const std::string& matrix_name) {
    bool pass = true;
    for (long i = 0; i < static_cast<long>(N_rows) * N_cols; ++i) {
        if (std::fabs(ref_C[i] - C_to_verify[i]) > EPSILON * std::max(1.0f, std::fabs(ref_C[i]))) {
            // Optionally print detailed errors to std::cerr if needed for debugging, 
            // but keep primary output to outFile as requested.
            // For example:
            // if (pass) { 
            //     std::cerr << "Verification FAILED for " << matrix_name << " (Matrix Size: " << N_rows << "x" << N_cols << ")" << std::endl;
            // }
            // std::cerr << "Mismatch at index " << i << " (Row " << i / N_cols << ", Col " << i % N_cols << "): CPU=" << ref_C[i] << ", GPU=" << C_to_verify[i]
            //           << ", Diff=" << std::fabs(ref_C[i] - C_to_verify[i]) << std::endl;
            pass = false;
            // return false; 
        }
    }
    return pass;
}

// GPU Warmup: Run a dummy kernel or operation to initialize GPU state [cite: 12]
// Modified to accept ostream to direct its output
void gpuWarmup(std::ostream& out) {
    out << "Performing GPU warmup..." << std::endl;
    float* dummy_d;
    size_t dummy_size = 1024 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dummy_d, dummy_size));
    
    float* dummy_h = new float[1024];
    for(int i=0; i<1024; ++i) dummy_h[i] = 0.f; 

    CUDA_CHECK(cudaMemcpy(dummy_d, dummy_h, dummy_size, cudaMemcpyHostToDevice));
        
    CUDA_CHECK(cudaFree(dummy_d));
    delete[] dummy_h;
    CUDA_CHECK(cudaDeviceSynchronize()); 
    out << "GPU warmup complete." << std::endl << std::endl;
}

// Prints some CUDA device properties
// Modified to accept ostream to direct its output
void printDeviceProperties(std::ostream& out) { // [cite: 17]
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA-enabled devices found!" << std::endl; // Errors still to cerr
        // outFile might not be open or valid if this happens early.
        // Or, if outFile is passed and valid: out << "No CUDA-enabled devices found!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device)); 
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    out << "--- Hardware and Runtime Environment (Partial) ---" << std::endl;
    out << "GPU Model: " << prop.name << std::endl;
    out << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    out << "Shared Memory Per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    out << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    out << "Max Threads Dim: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    out << "Max Grid Dim: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    out << "Warp Size: " << prop.warpSize << std::endl;
    
    int runtimeVersion, driverVersion;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    out << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 1000) / 10 << std::endl;
    out << "CUDA Driver Version: " << driverVersion / 1000 << "." << (driverVersion % 1000) / 10 << std::endl;
    out << "-------------------------------------------------" << std::endl << std::endl;
}