// Compile by 'nvcc -o 01_vector_addition 01_vector_addition.cu'
/*
Solving vector addition. Vector addition is a data parallel operation. Our dataset consists of three arrays: A, B, and C. The same operation is performed on each element:
Cx = Ax + Bx.

Memory allocation on GPU: CPU memory and GPU memory are physically separate memory. malloc allocates memory on the CPU's RAM. The GPU kernel/device function can only 
access memory that's allocated/pointing to the device memory. To allocate memory on the GPU, we need to use the cudaMalloc API. Unlike the malloc command, cudaMalloc 
does not return a pointer to allocated memory; instead, it takes a pointer reference as a parameter and updates the same with the allocated memory.

Transfer data from host memory to device memory w cudaMemcpy. Like other memcopy commands, this API requires the destination pointer, source pointer, and size. One 
additional parameter it takes is the direction of copy, that is, whether we are copying from the host to the device or from the device to the host. In the latest version 
of CUDA, this is optional since the driver is capable of understanding whether the pointer points to the host memory or device memory. Note that there is an
asynchronous alternative to cudaMemcpy.

Call and execute a CUDA function  <<<,>>>

Synchronize: As we mentioned in the Hello World program, kernel calls are asynchronous in nature. In order for the host to make sure that kernel execution
has finished, the host calls the cudaDeviceSynchronize function. This makes sure that all of the previously launched device calls have finished.
GPU will be stuck 100% if you do not have this line!

Transfer data from host memory to device memory: Use the same cudaMemcpy API to copy the data back from the device to the host for post-processing or
validation duties such as printing.

Free the allocated GPU memory: Finally, free the allocated GPU memory using the cudaFree API.

With a combination of threads and blocks, the unique ID of a thread can be calculated. As shown in the preceding code, another variable is given to all threads. 
This is called blockDim. This variable consists of the block's dimensions, that is, the number of threads per block.
int index = threadIdx.x + blockIdx.x * blockDim.x;  / calculate unique thread index
c[index] = a[index] + b[index];                     / access data on the thread

 NVIDIA Pascal card allows a
maximum of 1,024 threads per thread block in the x and y dimensions, while in the z
dimension, you can only launch 64 threads. Similarly, the maximum blocks in a grid are
restricted to 65,535 in the y and z dimensions in the Pascal architecture and 2^31 -1 in the
x dimension. If the developer launches a kernel with an unsupported dimension, the
application throws a runtime error.

*/

/*
// sequential code down is changed to good CUDA code described above
#include<stdio.h>
#include<stdlib.h>

#define N 1000000   // completes in 0m3.391s on CPU

void host_add(int *a, int *b, int *c) {
for(int idx=0;idx<N;idx++)
c[idx] = a[idx] + b[idx];
}

//basically just fills the array with index.
void fill_array(int *data) {
    for(int idx=0;idx<N;idx++)
    data[idx] = idx;
}

void print_output(int *a, int *b, int*c) {
    for(int idx=0;idx<N;idx++)
    printf("\n %d + %d = %d", a[idx] , b[idx], c[idx]);
}

int main(void) {
    int *a, *b, *c;
    int size = N * sizeof(int);
    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); fill_array(a);
    b = (int *)malloc(size); fill_array(b);
    c = (int *)malloc(size);
    host_add(a,b,c);
    print_output(a,b,c);
    free(a); free(b); free(c);
    return 0;
}
*/

#include <stdio.h>
#include <stdlib.h>

#define N 1000000

// GTX 1070 and 1650 no of blocks and threds
int threads_per_block = 64;
int no_of_blocks = 1024;

__global__ void device_add(int *a, int *b, int *c) {
    // If I do printf of results here, it hammers Nvidia GPU to 100% using #define N 10 000 000.
    // when changing to device_add<<<N,1>>> it's the same speed as printing inside 'void print_output(int *a, int *b, int*c)'
    // <<<no_of_blocks,threads_per_block>>> the fastest. Faster than 'void print_output(int *a, int *b, int*c)'
    // Time 21s b8 t1000000/8
    // Time 17s b8 t1000000
    // Time 3.2s b64 t1024 what GTX 1070 and 1650 have !!!! So printing here is much faster with GPU at 100% !!!
    // Time 0.4s b64 t1024 what GTX 1070 and 1650 have !!!! So printing here is much faster with GPU at 100% !!! If no printing
    /* 
    With a combination of threads and blocks, the unique ID of a thread can be calculated. As shown in the preceding code, another variable is given to all threads. 
    This is called blockDim. This variable consists of the block's dimensions, that is, the number of threads per block.
    */
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
    printf("\n %d + %d = %d", a[index], b[index], c[index]);
}

//basically just fills the array with index.
void fill_array(int *data) {
    for(int idx=0;idx<N;idx++)
        data[idx] = idx;
}

void print_output(int *a, int *b, int*c) {
    // If I do printf of results here, GPU is working in a spike to 25%. Using #define N 10 000 000 and device_add<<<N,1>>>.
    // When using device_add<<<1,1>>> one block and one thread, CPU [code above] is a tiny bit faster
    // device_add<<<N,1>>> will execute the device_add function N times in parallel instead of once. Each parallel invocation of the 
    // device_add function is referred to as a block.
    // device_add<<<1,N>>> is a bit faster here
    // device_add<<<N,N>>> faster than device_add<<<1,N>>>
    // <<<no_of_blocks,threads_per_block>>> slower than N,N
    // Time 29s b64 t1024 what GTX 1070 and 1650 have !!!! So printing above is much faster with GPU at 100% !!!
    //for(int idx=0;idx<N;idx++)
      // printf("\n %d + %d = %d", a[idx] , b[idx], c[idx]);
}

int main(void) {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);

    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); fill_array(a);
    b = (int *)malloc(size); fill_array(b);
    c = (int *)malloc(size);

    // Alloc space for device copies of vector (a, b, c)
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_b, N * sizeof(int));
    cudaMalloc((void **)&d_c, N * sizeof(int));

    // Copy from host to device
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N* sizeof(int), cudaMemcpyHostToDevice);

    
    device_add<<<no_of_blocks,threads_per_block>>>(d_a,d_b,d_c);

    // Sync GPU and host, so the host knows when the GPU is done. GPU will be stuck 100% if you do not have this line. If stuck 'sudo kill -9 pid' nvidia-smi shows pid's
    cudaDeviceSynchronize();

    // Copy result back to host. cudaMalloc allocates the data on the global device memory RAM. 
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    print_output(a,b,c);
    free(a); free(b); free(c);

    //free gpu memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}