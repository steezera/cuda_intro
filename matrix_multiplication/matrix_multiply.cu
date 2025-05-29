// 파일: matrix_multiply.cu
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <fstream>

// 테스트할 행렬 크기 정의
const int MATRIX_SIZES[] = {4, 16, 32, 64, 256};
const int NUM_SIZES = 5;
const int NUM_RUNS = 20;
const float EPSILON = 1e-5;
const int TILE_SIZE = 16;

// 직렬 CPU 행렬 곱셈
void matrixMulSerial(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Naive CUDA 커널
__global__ void matrixMulNaive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled CUDA 커널
__global__ void matrixMulTiled(float* A, float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 공유 메모리에 타일 로드
        if (row < N && t * TILE_SIZE + threadIdx.x < N) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < N) {
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // 부분 합 계산
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// 결과 검증
bool verifyResults(float* ref, float* test, int N) {
    for (int i = 0; i < N * N; i++) {
        if (fabs(ref[i] - test[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

// 행렬을 무작위 값으로 초기화
void initializeMatrix(float* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    // 결과 파일 열기
    std::ofstream outFile("results.txt");
    if (!outFile.is_open()) {
        std::cerr << "results.txt 파일을 열 수 없습니다.\n";
        return 1;
    }
    
    // 하드웨어 정보 출력
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // outFile << "GPU: " << prop.name << "\n";
    // outFile << "CUDA 버전: " << CUDART_VERSION / 1000 << "." << (CUDART_VERSION % 1000) / 10 << "\n";
    // outFile << "OS: " << __linux__ ? "Linux" : "Other" << "\n\n";
    
    // 타이밍 변수
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int s = 0; s < NUM_SIZES; s++) {
        int N = MATRIX_SIZES[s];
        outFile << "행렬 크기: " << N << "x" << N << "\n";
        
        // 호스트 메모리 할당
        size_t size = static_cast<size_t>(N) * N * sizeof(float); // 명시적 캐스팅 추가
        float *h_A = (float*)malloc(size);
        float *h_B = (float*)malloc(size);
        float *h_C_serial = (float*)malloc(size);
        float *h_C_naive = (float*)malloc(size);
        float *h_C_tiled = (float*)malloc(size);
        
        // 메모리 할당 확인
        if (!h_A || !h_B || !h_C_serial || !h_C_naive || !h_C_tiled) {
            std::cerr << "호스트 메모리 할당 실패\n";
            outFile.close();
            return 1;
        }
        
        // 행렬 초기화
        initializeMatrix(h_A, N);
        initializeMatrix(h_B, N);
        
        // 디바이스 메모리 할당
        float *d_A, *d_B, *d_C;
        cudaError_t err;
        err = cudaMalloc(&d_A, size);
        if (err != cudaSuccess) {
            std::cerr << "d_A 할당 실패: " << cudaGetErrorString(err) << "\n";
            outFile.close();
            return 1;
        }
        err = cudaMalloc(&d_B, size);
        if (err != cudaSuccess) {
            std::cerr << "d_B 할당 실패: " << cudaGetErrorString(err) << "\n";
            outFile.close();
            return 1;
        }
        err = cudaMalloc(&d_C, size);
        if (err != cudaSuccess) {
            std::cerr << "d_C 할당 실패: " << cudaGetErrorString(err) << "\n";
            outFile.close();
            return 1;
        }
        
        // 입력을 디바이스로 복사
        err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "d_A 복사 실패: " << cudaGetErrorString(err) << "\n";
            outFile.close();
            return 1;
        }
        err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "d_B 복사 실패: " << cudaGetErrorString(err) << "\n";
            outFile.close();
            return 1;
        }
        
        // 직렬 CPU 타이밍
        double serial_time = 0.0;
        for (int i = 0; i < NUM_RUNS; i++) {
            auto start_time = std::chrono::high_resolution_clock::now();
            matrixMulSerial(h_A, h_B, h_C_serial, N);
            auto end_time = std::chrono::high_resolution_clock::now();
            serial_time += std::chrono::duration<double, std::milli>(end_time - start_time).count();
        }
        serial_time /= NUM_RUNS;
        
        // Naive CUDA 타이밍
        float naive_time = 0.0f;
        dim3 threadsNaive(TILE_SIZE, TILE_SIZE);
        dim3 blocksNaive((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
        
        // 워밍업
        matrixMulNaive<<<blocksNaive, threadsNaive>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        
        for (int i = 0; i < NUM_RUNS; i++) {
            cudaEventRecord(start);
            matrixMulNaive<<<blocksNaive, threadsNaive>>>(d_A, d_B, d_C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            naive_time += ms;
        }
        naive_time /= NUM_RUNS;
        
        err = cudaMemcpy(h_C_naive, d_C, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "h_C_naive 복사 실패: " << cudaGetErrorString(err) << "\n";
            outFile.close();
            return 1;
        }
        
        // Tiled CUDA 타이밍
        float tiled_time = 0.0f;
        dim3 threadsTiled(TILE_SIZE, TILE_SIZE);
        dim3 blocksTiled((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
        
        // 워밍업
        matrixMulTiled<<<blocksTiled, threadsTiled>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        
        for (int i = 0; i < NUM_RUNS; i++) {
            cudaEventRecord(start);
            matrixMulTiled<<<blocksTiled, threadsTiled>>>(d_A, d_B, d_C, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            tiled_time += ms;
        }
        tiled_time /= NUM_RUNS;
        
        err = cudaMemcpy(h_C_tiled, d_C, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "h_C_tiled 복사 실패: " << cudaGetErrorString(err) << "\n";
            outFile.close();
            return 1;
        }
        
        // 결과 검증
        bool naive_correct = verifyResults(h_C_serial, h_C_naive, N);
        bool tiled_correct = verifyResults(h_C_serial, h_C_tiled, N);
        
        // 결과 출력
        outFile << "직렬 시간: " << serial_time << " ms\n";
        outFile << "Naive CUDA 시간: " << naive_time << " ms\n";
        outFile << "Tiled CUDA 시간: " << tiled_time << " ms\n";
        outFile << "Naive 정확성: " << (naive_correct ? "예" : "아니오") << "\n";
        outFile << "Tiled 정확성: " << (tiled_correct ? "예" : "아니오") << "\n";
        outFile << "직렬/Naive 속도 향상: " << serial_time / naive_time << "\n";
        outFile << "직렬/Tiled 속도 향상: " << serial_time / tiled_time << "\n";
        outFile << "Naive/Tiled 속도 향상: " << naive_time / tiled_time << "\n\n";
        
        // 메모리 해제
        free(h_A); free(h_B); free(h_C_serial); free(h_C_naive); free(h_C_tiled);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    outFile.close();
    
    std::cout << "결과가 results.txt에 저장되었습니다.\n";
    
    return 0;
}