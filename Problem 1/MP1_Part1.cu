// Kayne Lee, 20350003

#include <cuda_runtime.h>
#include <stdio.h>

int getNumCores(int major, int minor) {

    if (major == 1) {
        // tesla
        return 8;
    } else if (major == 2) {
        if (minor == 0) {
            // fermi generation
            return 32;
        } else {
            return 48;
        }
    } else if (major == 3) {
        // Kepler
        return 192;
    } else if (major == 5) {
        // Maxwell
        return 128;
    } else if (major == 6) {
        // Pascal
        if (minor == 0 || minor == 2) {
            return 64;
        } else if (minor == 1) {
            return 128;
        }
    } else if (major == 7) {
        // Volta and Turing
        if (minor == 0) {
            return 64;
        } else if (minor == 5) {
            return 64;
        }
    } else if (major == 8) { 
        // Ampere
        if (minor == 0 || minor == 6) {
            return 64;
        }
        else if (minor == 6) {
            return 128;
        }
    }
    // unknown architectures
    return -1; 
}

int main() {
    int numDevices;
    cudaGetDeviceCount(&numDevices);

    printf("Number of CUDA Devices: %d\n", numDevices);

    for (int i = 0; i < numDevices; i++) {
        // initialize and get the device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // calculate core count
        int nCores = getNumCores(prop.major, prop.minor) * prop.multiProcessorCount;

        printf("Device Number: %d\n", i);
        printf("-Device Name: %s\n", prop.name);
        printf("-Clock Rate: %d kHz\n", prop.clockRate);
        printf("-Number of streaming multiprocessors: %d\n", prop.multiProcessorCount);
        printf("-Number of cores: %d\n", nCores);
        printf("-Warp size: %d\n", prop.warpSize);
        printf("-Amount of global Memory: %.2f GB\n", (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("-Amount of constant Memory: %.2f KB\n", (float)prop.totalConstMem / 1024);
        printf("-Amount of shared Memory per Block: %.2f KB\n", (float)prop.sharedMemPerBlock / 1024);
        printf("-Number of registers available per Block: %d\n", prop.regsPerBlock);
        printf("-Maximum number of threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("-Maximum size pf each dimension of a block: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("-Maximum size of each dimenson of a grid: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    }

    return 0;
}
