//fixed 4 step reduce
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

//flexible step reduce 
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

//reduce using local memory (so called privatisation)
kernel void reduce_add_3(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}

//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
kernel void reduce_add_4(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
} #include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
    std::cerr << "Application usage:" << std::endl;
    std::cerr << "  -p : select platform " << std::endl;
    std::cerr << "  -d : select device" << std::endl;
    std::cerr << "  -l : list all platforms and devices" << std::endl;
    std::cerr << "  -f : input image file (default: test.pgm)" << std::endl;
    std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
    int platform_id = 0;
    int device_id = 0;
    string image_filename = "test.pgm";

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
        else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
        else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
        else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
    }

    cimg::exception_mode(0);
    try {
        // Load and display input image
        CImg<unsigned char> image_input(image_filename.c_str());
        CImgDisplay disp_input(image_input, "Input Image");

        // Handle input image channels
        CImg<unsigned char> gray_image;
        int channels = image_input.spectrum();
        std::cout << "Input image has " << channels << " channel(s)" << std::endl;

        if (channels == 1) {
            gray_image = image_input;
        } else if (channels == 3) {
            gray_image = image_input.get_RGBtoYCbCr().get_channel(0);
        } else {
            throw std::runtime_error("Unsupported number of channels. Expected 1 (grayscale) or 3 (RGB).");
        }

        size_t image_size = gray_image.size();
        const int num_bins = 256;

        // OpenCL setup
        // Handle input image channels
        cl::Context context = GetContext(platform_id, device_id);
        std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

        // Create command queue with profiling enabled
        cl::CommandQueue queue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], CL_QUEUE_PROFILING_ENABLE);

        cl::Program::Sources sources;
        AddSources(sources, "kernels/my_kernels.cl");
        cl::Program program(context, sources);
        try {
            program.build();
        } catch (const cl::Error& err) {
            std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw err;
        }

        // Device buffers
        cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_size);
        cl::Buffer dev_image_output(context, CL_MEM_WRITE_ONLY, image_size);
        cl::Buffer dev_histogram(context, CL_MEM_READ_WRITE, num_bins * sizeof(int));
        cl::Buffer dev_num_bins(context, CL_MEM_READ_ONLY, sizeof(int));

        // Timing setup using cl::Event
        cl::Event event;
        double mem_time = 0, kernel_time = 0;

        // Memory transfers to device (non-blocking)
        queue.enqueueWriteBuffer(dev_image_input, CL_FALSE, 0, image_size, gray_image.data(), nullptr, &event);
        event.wait(); // Wait for completion
        mem_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        queue.enqueueWriteBuffer(dev_num_bins, CL_FALSE, 0, sizeof(int), &num_bins, nullptr, &event);
        event.wait();
        mem_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        queue.enqueueFillBuffer(dev_histogram, 0, 0, num_bins * sizeof(int), nullptr, &event);
        event.wait();
        mem_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        // Step 1: Compute histogram
        cl::Kernel hist_kernel(program, "hist_simple_uchar");
        hist_kernel.setArg(0, dev_image_input);
        hist_kernel.setArg(1, dev_histogram);
        hist_kernel.setArg(2, dev_num_bins);
        queue.enqueueNDRangeKernel(hist_kernel, cl::NullRange, cl::NDRange(image_size), cl::NDRange(256), nullptr, &event);
        event.wait();
        kernel_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        // Step 2: Compute cumulative histogram
        cl::Kernel scan_kernel(program, "scan_bl");
        scan_kernel.setArg(0, dev_histogram);
        queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange, cl::NDRange(num_bins), cl::NDRange(256), nullptr, &event);
        event.wait();
        kernel_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        // Step 3 & 4: Normalize and back-project
        cl::Kernel backproj_kernel(program, "back_project");
        backproj_kernel.setArg(0, dev_image_input);
        backproj_kernel.setArg(1, dev_image_output);
        backproj_kernel.setArg(2, dev_histogram);
        backproj_kernel.setArg(3, (int)image_size);
        queue.enqueueNDRangeKernel(backproj_kernel, cl::NullRange, cl::NDRange(image_size), cl::NDRange(256), nullptr, &event);
        event.wait();
        kernel_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        // Memory transfer back to host (non-blocking)
        std::vector<unsigned char> output_buffer(image_size);
        queue.enqueueReadBuffer(dev_image_output, CL_FALSE, 0, image_size, output_buffer.data(), nullptr, &event);
        event.wait();
        mem_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        // Approximate total time as sum of measured times
        double total_time = mem_time + kernel_time;

        // Display timings
        std::cout << "Memory Transfer Time: " << mem_time << " ms" << std::endl;
        std::cout << "Kernel Execution Time: " << kernel_time << " ms" << std::endl;
        std::cout << "Total Execution Time (approx): " << total_time << " ms" << std::endl;

        // Display output image

        // Convert the output buffer into a CImg image and display it
        CImg<unsigned char> image_output(output_buffer.data(), gray_image.width(), gray_image.height());
        CImgDisplay disp_output(image_output, "Output Image");

        // Optionally save the output image
        image_output.save("output.pgm");

        // Keep windows open until closed
        while (!disp_input.is_closed() && !disp_output.is_closed()) {
            cimg::wait(20);
        }

		

    } catch (cl::Error& err) {
        std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
        if (err.err() == CL_PROFILING_INFO_NOT_AVAILABLE) {
            std::cerr << "Profiling info not available. Ensure your device supports profiling and CL_QUEUE_PROFILING_ENABLE is set." << std::endl;
        }
        return 1;
    } catch (CImgException& err) {
        std::cerr << "ERROR: " << err.what() << std::endl;
        return 1;
    } catch (const std::runtime_error& err) {
        std::cerr << "ERROR: " << err.what() << std::endl;
        return 1;
    }
}

//a very simple histogram implementation
kernel void hist_simple(global const int* A, global int* H) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//Hillis-Steele basic inclusive scan
//requires additional buffer B to avoid data overwrite 
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

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
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

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

//Blelloch basic exclusive scan
kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N-1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

//calculates the block sums
kernel void block_sum(global const int* A, global int* B, int local_size) {
	int id = get_global_id(0);
	B[id] = A[(id+1)*local_size-1];
}

//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void scan_add_atomic(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N; i++)
		atomic_add(&B[i], A[id]);
}

//adjust the values stored in partial scans by adding block sums to corresponding blocks
kernel void scan_add_adjust(global int* A, global const int* B) {
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
}