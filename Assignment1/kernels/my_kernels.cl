// Histogram kernel using local memory for 16-bit input with variable bins
kernel void hist_local(global const ushort* A, global int* H, int nr_bins, local int* local_hist) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    // Initialize local histogram
    for (int i = lid; i < nr_bins; i += local_size) {
        local_hist[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate histogram
    if (id < get_global_size(0)) {
        ushort value = A[id];
        int bin_index = (int)(((float)value / 65535.0f) * (nr_bins - 1)); // Scale to 0 to nr_bins-1
        atomic_inc(&local_hist[bin_index]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Merge to global histogram
    for (int i = lid; i < nr_bins; i += local_size) {
        atomic_add(&H[i], local_hist[i]);
    }
}

// Blelloch scan kernel
kernel void scan_bl(global int* A, const int nr_bins) {
    int id = get_global_id(0);
    if (id >= nr_bins) return;
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
        A[N - 1] = 0;
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

    for (int stride = 1; stride < nr_bins; stride *= 2) {
        int temp = 0;
        if (id >= stride) {
            temp = A[id - stride];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (id >= stride) {
            A[id] += temp;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Shift for exclusive scan
    if (id == 0) {
        for (int i = nr_bins - 1; i > 0; i--) {
            A[i] = A[i - 1];
        }
        A[0] = 0;
    }
}

// Normalize LUT kernel
kernel void normalize_lut(global const int* cum_histogram, global ushort* lut, float scale, const int nr_bins) {
    int id = get_global_id(0);
    if (id >= 65536) return;
    int bin = (int)(((float)id / 65535.0f) * (nr_bins - 1)); // Map to 0 to nr_bins-1
    lut[id] = (ushort)(cum_histogram[bin] * scale);
}

// Back projection kernel
kernel void back_project(global const ushort* input, global ushort* output, global ushort* lut) {
    int id = get_global_id(0);
    output[id] = lut[input[id]];
}