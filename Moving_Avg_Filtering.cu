#include <stdio.h>
#include <cuda.h>
#include <time.h>

struct Startup{
    int seed = time(nullptr);    
    int threadsPerBlock = 256;
    int datasetSize = 10000;
	int range = 100;
} startup;

struct DataSet{
    float* values;
    int  size;
};


inline int sizeOfDataSet(DataSet data)
{ return sizeof(float)*data.size; }

DataSet generateRandomDataSet(int size){
    DataSet data;
    data.size = size;
    data.values = (float*)malloc(sizeof(float)*data.size);

    for (int i = 0; i < data.size; i++)
        data.values[i] = (float)(rand()%startup.range);

    return data;
}

bool CompareDataSet(DataSet d1, DataSet d2){

    for (int i = 0; i < d1.size; i++)
        if (d1.values[i] != d2.values[i]){
            printf("Dataset is different");
            return false;
        }
        if (d1.size != d2.size) {printf("Datasets are not equal size\n"); return false;};
    	printf("D1 and D2 are equal!");
    	return true;

}

__global__ void DeviceCalculateSM_Global(float* input, int input_size, float* result, int result_size, int sample_size){
    int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	float sum = 0;
    if (id_x < result_size){

        
        for (int i = 0; i < sample_size; i++)
            sum = sum + input[id_x+i];
           sum = sum/sample_size;

        result[id_x] = sum;
    }
}

__global__ void DeviceCalculateSM_Shared(float* input, int input_size, float* result, int result_size, int sample_size){
    int id_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (id_x < input_size){

        extern __shared__ float cache[];

        int cachedSize = sample_size + blockDim.x;

        for (int i = 0; i < cachedSize/blockDim.x+1; i++){
            int cacheId = threadIdx.x+ i*blockDim.x;
            if (cacheId < cachedSize && cacheId+blockDim.x *blockIdx.x < input_size)
                cache[cacheId] = input[cacheId+blockDim.x *blockIdx.x];
        }
        __syncthreads();

        float sum = 0;
        for (int i = 0; i < sample_size; i++){
            if(i + threadIdx.x < cachedSize && i + id_x < input_size)
                sum = sum + cache[i+threadIdx.x];
        }
        sum = sum/sample_size;

        /*store in global memory*/
        if (id_x < result_size)
            result[id_x] = sum;
    }

}

DataSet CalculateSM(DataSet input, int sample_size, bool usesharedmemory){
    if(sample_size == 1 && input.size < 1 && sample_size < 1 && sample_size > input.size) 
	 { 
	 	printf("Error! Invalid Sample Size"); 
	 	exit(-1); 
	 }
    
    int result_size = input.size-sample_size+1;
    DataSet host_result = {(float*)malloc(sizeof(float)*(result_size)), result_size};

    float* device_input, *device_result;


    int threads_needed = host_result.size;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    if (usesharedmemory){
        int shared_memory_allocation_size = sizeof(float)*(startup.threadsPerBlock+sample_size);
        cudaEventRecord(start);
        DeviceCalculateSM_Shared<<<threads_needed/ startup.threadsPerBlock + 1, startup.threadsPerBlock, shared_memory_allocation_size>>> (device_input, input.size, device_result, host_result.size, sample_size);
        cudaEventRecord(stop);

    }else{
        cudaEventRecord(start);
        DeviceCalculateSM_Global<<<threads_needed/ startup.threadsPerBlock + 1, startup.threadsPerBlock>>> (device_input, input.size, device_result, host_result.size, sample_size);
        cudaEventRecord(stop);
    }

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    if (usesharedmemory) printf("Shared Memory: "); else printf("Global Memory: ");
    printf("Kernel executed in %f milliseconds\n", milliseconds);

    return host_result;
}

void printDataSet(DataSet data){
    for (int i = 0; i < data.size; i++)
        printf("%.6g, ", data.values[i]);
    printf("\n");
}


int main(int argc, char** argv){

    for (int i = 0; i < argc; i++){
        
        if (strcmp(argv[i],  "Range")==0 && i+1 < argc) startup.range = atoi(argv[i+1]);
        if (strcmp(argv[i],  "Seed")==0 && i+1 < argc) startup.seed = atoi(argv[i+1]);
        if (strcmp(argv[i],  "Block threads")==0 && i+1 < argc) startup.threadsPerBlock = atoi(argv[i+1]);
    }

    srand(startup.seed);

    DataSet data = generateRandomDataSet(100);
    printDataSet( data );
    DataSet shared = CalculateSM(data, 2, true);
    DataSet global = CalculateSM(data, 2, false);


    printDataSet( shared );
    printf("\n");
    printDataSet( global );
    printf("\n");


    printf("Each should be %d elements in size\n", global.size);
    CompareDataSet(global, shared);
}
