#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <float.h>

using namespace std;
using namespace chrono;

// Constants for the problem
const int NUM_CITIES = 29;
const int TOTAL_THREADS = 1024;
const int BLOCK_SIZE = 256;
const int NUM_BLOCKS = 4;
const int COMPARE_ITERATIONS = 100;
const double INITIAL_TEMPERATURE = 100.0;
const double TEMP_THRESHOLD = 1;
const double DECAY_RATE = 0.000025;

// Host-side function to calculate the total cost of a tour
double calculateCostHost(const int* tour, const double* graphWeights) {
    double totalCost = 0.0;
    for (int i = 0; i < NUM_CITIES; i++) {
        int u = tour[i];
        int v = tour[(i + 1) % NUM_CITIES]; // Wrap around for the last node
        totalCost += graphWeights[u * NUM_CITIES + v];
    }
    return totalCost;
}

// Device-side function to calculate the total cost of a tour
__device__ double calculateCost(const int* tour, const double* graphWeights) {
    double totalCost = 0.0;
    for (int i = 0; i < NUM_CITIES; i++) {
        int u = tour[i];
        int v = tour[(i + 1) % NUM_CITIES]; // Wrap around
        totalCost += graphWeights[u * NUM_CITIES + v];
    }
    return totalCost;
}

// Function to decide whether to accept a new solution based on probability
__device__ bool accept(double T, double deltaCost, curandState_t* state) {
    if (deltaCost < 0 || exp(-deltaCost / T) > curand_uniform(state)) {
        return true;
    }
    return false;
}

// Simple city swap function for mutation
__device__ void swapCities(int* tour, int city1, int city2) {
    int temp = tour[city1];
    tour[city1] = tour[city2];
    tour[city2] = temp;
}

// Kernel function for Simulated Annealing
__global__ void simulatedAnnealing(int(*tour)[NUM_CITIES], double* cost, const double* graphWeights, double initialTemperature, double threshold, double decayRate, unsigned int seed, int* globalTourMemory, int compareIterations, double* blockBestCosts) {
    __shared__ double threadCosts[BLOCK_SIZE]; // Holds the current cost of each thread's tour within a block
    __shared__ int threadIndices[BLOCK_SIZE]; // Holds the thread indices corresponding to each tour within a block.
    __shared__ int sharedBestTour[NUM_CITIES]; // Shared memory for the best tour in the block

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    // Initialize curandState_t for random number generation
    curandState_t state;
    curand_init(seed + tid, tid, 0, &state);

    // Initialize the shared initial tour for the block
    if (lid == 0) {
        for (int i = 0; i < NUM_CITIES; i++) {
            sharedBestTour[i] = tour[blockIdx.x * BLOCK_SIZE][i];
        }
    }
    __syncthreads();

    // Pointer to global memory for currentTour and bestTour
    int* currentTour = &globalTourMemory[tid * NUM_CITIES];

    // Initialize current tour and cost with the shared initial tour and cost
    for (int i = 0; i < NUM_CITIES; i++) {
        currentTour[i] = sharedBestTour[i];
    }
    double currentCost = calculateCost(currentTour, graphWeights);

    // Calculate temperature reduction points based on compareIterations
    double temperatureStep = (initialTemperature - threshold) / compareIterations;
    double nextReductionTemp = initialTemperature - temperatureStep;

    // Begin Simulated Annealing
    double T = initialTemperature;
    while (T > threshold) {
        // Generate a neighboring solution by swapping two cities
        int city1 = curand(&state) % NUM_CITIES;
        int city2 = curand(&state) % NUM_CITIES;
        while (city1 == city2) {
            city2 = curand(&state) % NUM_CITIES;
        }
        swapCities(currentTour, city1, city2);

        // Calculate the cost of the new solution
        double neighborCost = calculateCost(currentTour, graphWeights);

        // Determine whether to accept the new solution
        if (accept(T, neighborCost - currentCost, &state)) {
            currentCost = neighborCost;
        }
        else {
            // Revert the swap if not accepted
            swapCities(currentTour, city1, city2);
        }

        // Perform warp-level reduction at temperature reduction points
        if (T <= nextReductionTemp) {
            threadCosts[lid] = currentCost;
            threadIndices[lid] = tid;
            __syncthreads();

            // Warp-level reduction
            for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                if (lid < offset) {
                    if (threadCosts[lid] > threadCosts[lid + offset]) {
                        threadCosts[lid] = threadCosts[lid + offset];
                        threadIndices[lid] = threadIndices[lid + offset];
                    }
                }
                __syncthreads();
            }

            // Update the best tour in the block
            if (lid == 0) {
                blockBestCosts[blockIdx.x] = threadCosts[0];
                for (int i = 0; i < NUM_CITIES; i++) {
                    sharedBestTour[i] = globalTourMemory[threadIndices[0] * NUM_CITIES + i];
                }
            }
            __syncthreads();

            // Copy the best tour in the block to all threads' current tours
            for (int i = 0; i < NUM_CITIES; i++) {
                currentTour[i] = sharedBestTour[i];
            }
            currentCost = calculateCost(currentTour, graphWeights);

            // Update next temperature reduction point
            nextReductionTemp -= temperatureStep;
        }

        // Decrease temperature
        T -= decayRate;
    }

    // Store the final cost for each thread
    cost[tid] = currentCost;
}

// Function to read graph weights from a file
void readGraphWeights(double* graphWeights, const string& filename) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        exit(1);
    }

    for (int i = 0; i < NUM_CITIES * NUM_CITIES; i++) {
        infile >> graphWeights[i];
    }
    infile.close();
}

int main() {
    // Read graph weights from file
    double* graphWeights;
    cudaMallocManaged(&graphWeights, NUM_CITIES * NUM_CITIES * sizeof(double));
    readGraphWeights(graphWeights, "matrix/bays29.txt");

    // Generate random initial tour for each block
    int initialTours[NUM_BLOCKS][NUM_CITIES];
    for (int b = 0; b < NUM_BLOCKS; b++) {
        for (int i = 0; i < NUM_CITIES; i++) {
            initialTours[b][i] = i; // Assign cities in order
        }
        srand(static_cast<unsigned int>(time(nullptr)) + b);
        random_shuffle(initialTours[b], initialTours[b] + NUM_CITIES);
    }

    // Allocate memory for tour and cost on device
    int(*tour)[NUM_CITIES];
    cudaMallocManaged(&tour, TOTAL_THREADS * NUM_CITIES * sizeof(int));
    for (int b = 0; b < NUM_BLOCKS; b++) {
        for (int t = 0; t < BLOCK_SIZE; t++) {
            for (int j = 0; j < NUM_CITIES; j++) {
                tour[b * BLOCK_SIZE + t][j] = initialTours[b][j];
            }
        }
    }

    double* cost;
    cudaMallocManaged(&cost, TOTAL_THREADS * sizeof(double));
    for (int b = 0; b < NUM_BLOCKS; b++) {
        double initialCost = calculateCostHost(initialTours[b], graphWeights);
        for (int t = 0; t < BLOCK_SIZE; t++) {
            cost[b * BLOCK_SIZE + t] = initialCost;
        }
    }

    // Allocate global memory for storing tours for each thread
    int* globalTourMemory;
    cudaMallocManaged(&globalTourMemory, 2 * TOTAL_THREADS * NUM_CITIES * sizeof(int));

    // Allocate memory for storing best costs for each block
    double* blockBestCosts;
    cudaMallocManaged(&blockBestCosts, NUM_BLOCKS * sizeof(double));

    // Timing the kernel execution
    auto start = high_resolution_clock::now();

    // Execute the kernel
    simulatedAnnealing << <NUM_BLOCKS, BLOCK_SIZE >> > (tour, cost, graphWeights, INITIAL_TEMPERATURE, TEMP_THRESHOLD, DECAY_RATE, 0, globalTourMemory, COMPARE_ITERATIONS, blockBestCosts);
    cudaDeviceSynchronize();

    auto end = high_resolution_clock::now();
    duration<double> duration = end - start;

    // Output the best solution found
    double bestCost = DBL_MAX;
    for (int i = 0; i < TOTAL_THREADS; i++) {
        if (cost[i] < bestCost) {
            bestCost = cost[i];
        }
    }
    cout << "Best cost: " << fixed << setprecision(2) << bestCost << endl;
    cout << "Time spent: " << duration.count() << " seconds" << endl;

    // Free memory
    cudaFree(tour);
    cudaFree(cost);
    cudaFree(globalTourMemory);
    cudaFree(blockBestCosts);
    cudaFree(graphWeights);

    return 0;
}
