//Kernels


kernel void identity(global const uchar* A, global uchar* B) {
    int id = get_global_id(0);
    B[id] = A[id];
}

kernel void filter_r(global const uchar* A, global uchar* B) {
    int id = get_global_id(0);
    int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
    int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

    //this is just a copy operation, modify to filter out the individual colour channels
    B[id] = A[id];
}

//identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
    int width = get_global_size(0); //image width in pixels
    int height = get_global_size(1); //image height in pixels
    int image_size = width*height; //image size in pixels
    int channels = get_global_size(2); //number of colour channels: 3 for RGB

    int x = get_global_id(0); //current x coord.
    int y = get_global_id(1); //current y coord.
    int c = get_global_id(2); //current colour channel

    int id = x + y*width + c*image_size; //global id in 1D space

    B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
    int width = get_global_size(0); //image width in pixels
    int height = get_global_size(1); //image height in pixels
    int image_size = width*height; //image size in pixels
    int channels = get_global_size(2); //number of colour channels: 3 for RGB

    int x = get_global_id(0); //current x coord.
    int y = get_global_id(1); //current y coord.
    int c = get_global_id(2); //current colour channel

    int id = x + y*width + c*image_size; //global id in 1D space

    uint result = 0;

    for (int i = (x-1); i <= (x+1); i++)
    for (int j = (y-1); j <= (y+1); j++) 
        result += A[i + j*width + c*image_size];

    result /= 9;

    B[id] = (uchar)result;
}

//convolution kernel, 2D
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
    int width = get_global_size(0); //image width in pixels
    int height = get_global_size(1); //image height in pixels
    int image_size = width*height; //image size in pixels
    int channels = get_global_size(2); //number of colour channels: 3 for RGB

    int x = get_global_id(0); //current x coord.
    int y = get_global_id(1); //current y coord.
    int c = get_global_id(2); //current colour channel

    int id = x + y*width + c*image_size; //global id in 1D space

    float result = 0;

    for (int i = (x-1); i <= (x+1); i++)
    for (int j = (y-1); j <= (y+1); j++) 
        result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];

    B[id] = (uchar)result;
}

kernel void rgb2grey(global const uchar* A, global uchar* B) {
    int id = get_global_id(0);
    int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
    int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

    if (colour_channel == 0) {
        B[id] = A[id] * 0.2126;
    } else 
    if (colour_channel == 1) {
        B[id] = A[id] * 0.7152;
    } else {
        B[id] = A[id] * 0.0722;
    }

    if (colour_channel == 0) {
        B[id] = 255 - ((A[id] * 0.2126) + (A[id + image_size] * 0.7152) + (A[id + (image_size * 2)] * 0.0722));
    } else 
    if (colour_channel == 1) {
        B[id] = 255 - ((A[id - image_size] * 0.2126) + (A[id] * 0.7152) + (A[id + image_size] * 0.0722));
    } else {
        B[id] = 255 - ((A[id - (image_size * 2)] * 0.2126) + (A[id - image_size] * 0.7152) + (A[id] * 0.0722));
    }
}

kernel void reduce_add_1(global const int* A, global int* B) {
    int id = get_global_id(0);
    int N = get_global_size(0);

    B[id] = A[id]; //copy input to output

    barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying
     
    //perform reduce on the output array
    //modulo operator is used to skip a set of values (e.g. 2 in the next line)
    //we also check if the added element is within bounds (i.e. < N)
    if (((id % 2) == 0) && ((id + 1) < N)) 
        B[id] += B[id + 1];

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (((id % 4) == 0) && ((id + 2) < N)) 
        B[id] += B[id + 2];

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (((id % 8) == 0) && ((id + 4) < N)) 
        B[id] += B[id + 4];

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (((id % 16) == 0) && ((id + 8) < N)) 
        B[id] += B[id + 8];
}


kernel void reduce_add_2(global const int* A, global int* B) {
    int id = get_global_id(0);
    int N = get_global_size(0);

    B[id] = A[id];

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) { //i is a stride
        if (!(id % (i * 2)) && ((id + i) < N)) 
            B[id] += B[id + i];

        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}


kernel void reduce_add_3(global const int* A, global int* B, local int* scratch) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    //cache all values from global to local memory
    scratch[lid] = A[id];

    barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

    for (int i = 1; i < N; i *= 2) {
        if (!(lid % (i * 2)) && ((lid + i) < N)) 
            scratch[lid] += scratch[lid + i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //oytput array
    B[id] = scratch[lid];
}


kernel void reduce_add_4(global const int* A, global int* B, local int* scratch) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);


    scratch[lid] = A[id];

    barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

    for (int i = 1; i < N; i *= 2) {
        if (!(lid % (i * 2)) && ((lid + i) < N)) 
            scratch[lid] += scratch[lid + i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }


   //output array
    if (!lid) {
        atomic_add(&B[0],scratch[lid]);
    }
}

//histogram equalization
kernel void hist_simple(global const uchar* A, global int* H, int nr_bins) {
    int id = get_global_id(0);
    int bin_index = A[id];
    if (bin_index < nr_bins) {
        atomic_inc(&H[bin_index]);
    }
}

//Hillis-Steele basic inclusive scan
//buffer B to avoid data overwrite 
kernel void scan_hs(global int* A, global int* B) {
    int id = get_global_id(0);
    int N = get_global_size(0);
    global int* C;

    for (int stride = 1; stride < N; stride *= 2) {
        B[id] = A[id];
        if (id >= stride)
            B[id] += A[id - stride];

        barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

        C = A; A = B; B = C; //swap A & B between steps
    }
}

//hillis-Steele inclusive scan, double bufferd
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);
    local int *scratch_3;//used for buffer swap

    //cache all N values from global memory to local memory
    scratch_1[lid] = A[id];

    barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

    for (int i = 1; i < N; i *= 2) {
        if (lid >= i)
            scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
        else
            scratch_2[lid] = scratch_1[lid];

        barrier(CLK_LOCAL_MEM_FENCE);

      
        scratch_3 = scratch_2;
        scratch_2 = scratch_1;
        scratch_1 = scratch_3;
    }

   
    B[id] = scratch_1[lid];
}

//scan for cumulative histogram
kernel void scan_bl(global int* A) {
    int id = get_global_id(0);
    int N = get_global_size(0);
    int t;

    //up
    for (int stride = 1; stride < N; stride *= 2) {
        if (((id + 1) % (stride*2)) == 0)
            A[id] += A[id - stride];

        barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
    }
    //down

    if (id == 0)
        A[N-1] = 0;//exclusive scan

    barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

    for (int stride = N/2; stride > 0; stride /= 2) {
        if (((id + 1) % (stride*2)) == 0) {
            t = A[id];
            A[id] += A[id - stride]; //reduce 
            A[id - stride] = t;      //move
        }

        barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
    }
}

//calculates block sums
kernel void block_sum(global const int* A, global int* B, int local_size) {
    int id = get_global_id(0);
    B[id] = A[(id+1)*local_size-1];
}

//A scan based on operations 
kernel void scan_add_atomic(global int* A, global int* B) {
    int id = get_global_id(0);
    int N = get_global_size(0);
    for (int i = id+1; i < N && id < N; i++)
        atomic_add(&B[i], A[id]);
}

//adjusting the values by adding block sums 
kernel void scan_add_adjust(global int* A, global const int* B) {
    int id = get_global_id(0);
    int gid = get_group_id(0);
    A[id] += B[gid];
}

// Normalizing for histogram equalization
kernel void normalize_lut(global const int* cum_histogram, global uchar* lut, float scale) {
    int id = get_global_id(0);
    lut[id] = (uchar)(cum_histogram[id] * scale);
}

// Back projection for histogram equalization
kernel void back_project(global const uchar* input, global uchar* output, global uchar* lut) {
    int id = get_global_id(0);
    output[id] = lut[input[id]];
}