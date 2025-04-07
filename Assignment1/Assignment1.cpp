#include <iostream>
#include <vector>
#include <algorithm>
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
    std::string image_filename = "test.pgm";

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
        else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
        else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
        else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
    }

    cimg::exception_mode(0);
    try {
        CImg<unsigned char> image_input(image_filename.c_str());
        CImgDisplay disp_input(image_input, "Input Image");

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

        cl::Context context = GetContext(platform_id, device_id);
        std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

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

        cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_size);
        cl::Buffer dev_image_output(context, CL_MEM_WRITE_ONLY, image_size);
        cl::Buffer dev_histogram(context, CL_MEM_READ_WRITE, num_bins * sizeof(int));
        cl::Buffer dev_num_bins(context, CL_MEM_READ_ONLY, sizeof(int));

        cl::Event event;
        double mem_time = 0, kernel_time = 0;

        queue.enqueueWriteBuffer(dev_image_input, CL_FALSE, 0, image_size, gray_image.data(), nullptr, &event);
        event.wait();
        mem_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        queue.enqueueWriteBuffer(dev_num_bins, CL_FALSE, 0, sizeof(int), &num_bins, nullptr, &event);
        event.wait();
        mem_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        queue.enqueueFillBuffer(dev_histogram, 0, 0, num_bins * sizeof(int), nullptr, &event);
        event.wait();
        mem_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        cl::Kernel hist_kernel(program, "hist_simple_uchar");
        hist_kernel.setArg(0, dev_image_input);
        hist_kernel.setArg(1, dev_histogram);
        hist_kernel.setArg(2, dev_num_bins);
        queue.enqueueNDRangeKernel(hist_kernel, cl::NullRange, cl::NDRange(image_size), cl::NDRange(256), nullptr, &event);
        event.wait();
        kernel_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        // === NEW: Read histogram back to host ===
        std::vector<int> histogram(num_bins);
        queue.enqueueReadBuffer(dev_histogram, CL_TRUE, 0, num_bins * sizeof(int), histogram.data());

        // === NEW: Display histogram using CImg ===
        const int hist_width = 512, hist_height = 400;
        CImg<unsigned char> hist_img(hist_width, hist_height, 1, 3, 255); // white
        const unsigned char black[] = { 0, 0, 0 };
        int max_value = *std::max_element(histogram.begin(), histogram.end());

        for (int i = 0; i < num_bins; ++i) {
            int x0 = i * (hist_width / num_bins);
            int x1 = (i + 1) * (hist_width / num_bins) - 1;
            int bar_height = static_cast<int>((histogram[i] / (float)max_value) * (hist_height - 10));
            hist_img.draw_rectangle(x0, hist_height - bar_height, x1, hist_height, black);
        }

        CImgDisplay disp_hist(hist_img, "Histogram");

        cl::Kernel scan_kernel(program, "scan_bl");
        scan_kernel.setArg(0, dev_histogram);
        queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange, cl::NDRange(num_bins), cl::NDRange(256), nullptr, &event);
        event.wait();
        kernel_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

		// === NEW: Read cumulative histogram from device ===
std::vector<int> cumulative_histogram(num_bins);
queue.enqueueReadBuffer(dev_histogram, CL_TRUE, 0, num_bins * sizeof(int), cumulative_histogram.data());

// === NEW: Display cumulative histogram (Fig. 1c) ===
CImg<unsigned char> cum_hist_img(hist_width, hist_height, 1, 3, 255);
int max_cum = cumulative_histogram.back(); // Last value = total pixel count
for (int i = 0; i < num_bins; ++i) {
    int x0 = i * (hist_width / num_bins);
    int x1 = (i + 1) * (hist_width / num_bins) - 1;
    int bar_height = static_cast<int>((cumulative_histogram[i] / (float)max_cum) * (hist_height - 10));
    cum_hist_img.draw_rectangle(x0, hist_height - bar_height, x1, hist_height, black);
}
CImgDisplay disp_cum_hist(cum_hist_img, "Cumulative Histogram");

// === NEW: Normalise & scale cumulative histogram (Fig. 1d) ===
std::vector<unsigned char> norm_cum_hist(num_bins);
for (int i = 0; i < num_bins; ++i) {
    norm_cum_hist[i] = static_cast<unsigned char>(255.0f * cumulative_histogram[i] / max_cum);
}

// === NEW: Display normalized cumulative histogram (Fig. 1d) ===
CImg<unsigned char> norm_hist_img(hist_width, hist_height, 1, 3, 255);
for (int i = 0; i < num_bins; ++i) {
    int x0 = i * (hist_width / num_bins);
    int x1 = (i + 1) * (hist_width / num_bins) - 1;
    int bar_height = static_cast<int>((norm_cum_hist[i] / 255.0f) * (hist_height - 10));
    norm_hist_img.draw_rectangle(x0, hist_height - bar_height, x1, hist_height, black);
}
CImgDisplay disp_norm_hist(norm_hist_img, "Normalized & Scaled Cumulative Histogram");


        cl::Kernel backproj_kernel(program, "back_project");
        backproj_kernel.setArg(0, dev_image_input);
        backproj_kernel.setArg(1, dev_image_output);
        backproj_kernel.setArg(2, dev_histogram);
        backproj_kernel.setArg(3, (int)image_size);
        queue.enqueueNDRangeKernel(backproj_kernel, cl::NullRange, cl::NDRange(image_size), cl::NDRange(256), nullptr, &event);
        event.wait();
        kernel_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        std::vector<unsigned char> output_buffer(image_size);
        queue.enqueueReadBuffer(dev_image_output, CL_FALSE, 0, image_size, output_buffer.data(), nullptr, &event);
        event.wait();
        mem_time += (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1e6;

        double total_time = mem_time + kernel_time;

        std::cout << "Memory Transfer Time: " << mem_time << " ms" << std::endl;
        std::cout << "Kernel Execution Time: " << kernel_time << " ms" << std::endl;
        std::cout << "Total Execution Time (approx): " << total_time << " ms" << std::endl;

        CImg<unsigned char> image_output(output_buffer.data(), gray_image.width(), gray_image.height());
        CImgDisplay disp_output(image_output, "Output Image");

        image_output.save("output.pgm");

        while (!disp_input.is_closed() && !disp_output.is_closed() && !disp_hist.is_closed()) {
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
