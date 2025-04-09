// Histogram kernel using local memory for 16-bit input with variable bins
kernel void hist_local(global const ushort* A, global int* H, int nr_bins, local int* local_hist) {
    int id = get_global_id(0);         // Global ID across all work-items
    int lid = get_local_id(0);         // Local ID within the work-group
    int group_id = get_group_id(0);    // Work-group ID
    int local_size = get_local_size(0); // Number of work-items in a work-group

    // Initialize local histogram to zero for this work-group
    if (lid < nr_bins) {
        local_hist[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Ensure all local memory is initialized

    // Calculate histogram for this work-item’s pixel
    if (id < get_global_size(0)) { // Guard against out-of-bounds access
        ushort value = A[id];
        int bin_index = (value * nr_bins) / 65536;
        if (bin_index >= nr_bins) bin_index = nr_bins - 1;
        atomic_inc(&local_hist[bin_index]);
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Synchronize within the work-group

    // Merge local histogram into global histogram
    if (lid < nr_bins) {
        atomic_add(&H[lid], local_hist[lid]);
    }
}
//SCANS 

// Blelloch basic exclusive scan for cumulative histogram with variable bins
kernel void scan_bl(global int* A, const int nr_bins) {
    int id = get_global_id(0);
    if (id >= nr_bins) return; // Guard against out-of-bounds access
    int N = nr_bins;
    int t;

    // Up-sweep
    for (int stride = 1; stride < N; stride *= 2) {
        if (((id + 1) % (stride * 2)) == 0)
            A[id] += A[id - stride];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Down-sweep
    if (id == 0)
        A[N - 1] = 0; // Exclusive scan
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int stride = N / 2; stride > 0; stride /= 2) {
        if (((id + 1) % (stride * 2)) == 0) {
            t = A[id];
            A[id] += A[id - stride];
            A[id - stride] = t;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
// Hillis-Steele scan kernel
kernel void scan_hs(global int* A, const int nr_bins) {
    int id = get_global_id(0);
    if (id >= nr_bins) return;

    // Hillis-Steele performs a series of shifts and additions
    for (int stride = 1; stride < nr_bins; stride *= 2) {
        int temp = 0;
        if (id >= stride) {
            temp = A[id - stride];
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // Synchronize before updating
        if (id >= stride) {
            A[id] += temp;
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // Synchronize after updating
    }

    // Shift right to make it exclusive (original values are inclusive)
    if (id == 0) {
        for (int i = nr_bins - 1; i > 0; i--) {
            A[i] = A[i - 1];
        }
        A[0] = 0;
    }
}


// Normalize LUT kernel for 16-bit output with variable bins
kernel void normalize_lut(global const int* cum_histogram, global ushort* lut, float scale, const int nr_bins) {
    int id = get_global_id(0);
    if (id >= 65536) return; // Guard against out-of-bounds access
    int bin = (id * nr_bins) / 65536; // Map 16-bit value to nr_bins
    if (bin >= nr_bins) bin = nr_bins - 1; // Clamp to valid range
    lut[id] = (ushort)(cum_histogram[bin] * scale); // Scale to 16-bit range
}

// Back projection kernel for 16-bit data
kernel void back_project(global const ushort* input, global ushort* output, global ushort* lut) {
    int id = get_global_id(0);
    output[id] = lut[input[id]];
}