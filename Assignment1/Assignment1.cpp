#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
    std::cerr << "Application usage:" << std::endl;
    std::cerr << "  -p : select platform " << std::endl;
    std::cerr << "  -d : select device" << std::endl;
    std::cerr << "  -l : list all platforms and devices" << std::endl;
    std::cerr << "  -f : input image file" << std::endl;
    std::cerr << "  -b : number of bins (1-1025, default 256)" << std::endl;
    std::cerr << "  -s : scan kernel (bl for Blelloch, hs for Hillis-Steele, default bl)" << std::endl;
    std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
    int platform_id = 0;
    int device_id = 0;
    char image_filename[256] = "mdr16.ppm"; // C-style string with reasonable size
    int num_bins = 256;
    char scan_kernel_type[3] = "bl"; // "bl" or "hs"

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i < argc - 1) { platform_id = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-d") == 0 && i < argc - 1) { device_id = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; return 0; }
        else if (strcmp(argv[i], "-f") == 0 && i < argc - 1) { strcpy(image_filename, argv[++i]); }
        else if (strcmp(argv[i], "-b") == 0 && i < argc - 1) { num_bins = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-s") == 0 && i < argc - 1) { strcpy(scan_kernel_type, argv[++i]); }
        else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
    }

    // Validate inputs
    if (num_bins < 1 || num_bins > 2500) {
        std::cerr << "Error: Number of bins must be between 1 and 1025" << std::endl;
        return 1;
    }
    if (strcmp(scan_kernel_type, "bl") != 0 && strcmp(scan_kernel_type, "hs") != 0) {
        std::cerr << "Error: Scan kernel must be 'bl' (Blelloch) or 'hs' (Hillis-Steele)" << std::endl;
        return 1;
    }

    cimg::exception_mode(0);

    try {
        // Load input image
        CImg<unsigned short> image_input;
        bool is_8bit = false;
        FILE* file = fopen(image_filename, "rb");
        if (!file) throw CImgIOException("Cannot open file");
        char magic[3] = {0};
        int maxval = 0;
        fscanf(file, "%2s %*d %*d %d", magic, &maxval);
        fclose(file);
        is_8bit = (maxval <= 255);

        if (is_8bit) {
            CImg<unsigned char> image_8bit(image_filename);
            image_input.assign(image_8bit.width(), image_8bit.height(), 1, image_8bit.spectrum());
            cimg_forXYC(image_input, x, y, c) {
                image_input(x, y, 0, c) = (unsigned short)(image_8bit(x, y, 0, c) * 257); // Scale 8-bit to 16-bit
            }
        } else {
            image_input = CImg<unsigned short>(image_filename);
        }

        CImgDisplay disp_input(image_input, "Input Image");

        // Setup OpenCL
        cl::Context context = GetContext(platform_id, device_id);
        std::cout << "Running on " << GetPlatformName(platform_id) << ", " 
                  << GetDeviceName(platform_id, device_id) << std::endl;
        cl::CommandQueue queue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], CL_QUEUE_PROFILING_ENABLE);

        // Load and build kernels
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

        // Image properties
        size_t width = image_input.width();
        size_t height = image_input.height();
        size_t channels = image_input.spectrum();
        size_t image_size = width * height;

        // Split into channels
        std::vector<CImg<unsigned short> > input_channels(channels);
        for (int c = 0; c < channels; c++) {
            input_channels[c] = CImg<unsigned short>(width, height, 1, 1);
            cimg_forXY(image_input, x, y) {
                input_channels[c](x, y) = image_input(x, y, 0, c);
            }
        }

        // Device buffers
        std::vector<cl::Buffer> dev_image_input(channels);
        std::vector<cl::Buffer> dev_image_output(channels);
        std::vector<cl::Buffer> dev_histogram(channels);
        std::vector<cl::Buffer> dev_cum_histogram(channels);
        std::vector<cl::Buffer> dev_lut(channels);

        for (int c = 0; c < channels; c++) {
            dev_image_input[c] = cl::Buffer(context, CL_MEM_READ_ONLY, image_size * sizeof(unsigned short));
            dev_image_output[c] = cl::Buffer(context, CL_MEM_WRITE_ONLY, image_size * sizeof(unsigned short));
            dev_histogram[c] = cl::Buffer(context, CL_MEM_READ_WRITE, num_bins * sizeof(unsigned int));
            dev_cum_histogram[c] = cl::Buffer(context, CL_MEM_READ_WRITE, num_bins * sizeof(unsigned int));
            dev_lut[c] = cl::Buffer(context, CL_MEM_READ_WRITE, 65536 * sizeof(unsigned short));
        }

        // Metrics structure
        struct StepMetrics {
            double transfer_time;
            double kernel_time;
            double total_time;
            size_t work;
            size_t span;
        };
        std::vector<std::vector<StepMetrics> > metrics(channels, std::vector<StepMetrics>(5));

        // Visualization displays
        std::vector<CImgDisplay> disp_hist(channels);
        std::vector<CImgDisplay> disp_cum_hist(channels);
        std::vector<CImgDisplay> disp_norm_cum_hist(channels);

        // Device properties
        cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        cl_ulong local_mem_size;
        device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &local_mem_size);
        size_t max_work_group_size;
        device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_work_group_size);
        std::cout << "Local Memory Size: " << local_mem_size << " bytes, Max Work-Group Size: " 
                  << max_work_group_size << std::endl;

        // Process each channel
        for (int c = 0; c < channels; c++) {
            // Step 1: Transfer input and initialize histogram
            cl::Event event1a, event1b;
            queue.enqueueWriteBuffer(dev_image_input[c], CL_TRUE, 0, image_size * sizeof(unsigned short), 
                                   input_channels[c].data(), NULL, &event1a);
            std::vector<unsigned int> zeros(num_bins, 0);
            queue.enqueueWriteBuffer(dev_histogram[c], CL_TRUE, 0, num_bins * sizeof(unsigned int), 
                                   zeros.data(), NULL, &event1b);
            event1a.wait();
            event1b.wait();
            metrics[c][0].transfer_time = (event1a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                                         event1a.getProfilingInfo<CL_PROFILING_COMMAND_START>() +
                                         event1b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                                         event1b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            metrics[c][0].total_time = metrics[c][0].transfer_time;
            metrics[c][0].work = image_size + num_bins;
            metrics[c][0].span = 1;

            // Step 2: Histogram calculation
            cl::Event event2a, event2b;
            cl::Kernel hist_kernel(program, "hist_local");
            hist_kernel.setArg(0, dev_image_input[c]);
            hist_kernel.setArg(1, dev_histogram[c]);
            hist_kernel.setArg(2, num_bins);
            size_t local_size = 1024;
            if (num_bins * sizeof(int) > local_mem_size) {
                std::cerr << "Error: Local histogram size (" << num_bins * sizeof(int) 
                          << " bytes) exceeds device local memory (" << local_mem_size << " bytes)" << std::endl;
                return 1;
            }
            if (local_size > max_work_group_size) {
                local_size = max_work_group_size;
            }
            hist_kernel.setArg(3, cl::Local(num_bins * sizeof(int)));
            size_t global_size = image_size;
            if (global_size % local_size != 0) {
                global_size = ((global_size / local_size) + 1) * local_size;
            }
            queue.enqueueNDRangeKernel(hist_kernel, cl::NullRange, cl::NDRange(global_size), 
                                       cl::NDRange(local_size), NULL, &event2a);
            event2a.wait();
            metrics[c][1].kernel_time = (event2a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                                       event2a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;

            std::vector<unsigned int> histogram(num_bins);
            queue.enqueueReadBuffer(dev_histogram[c], CL_TRUE, 0, num_bins * sizeof(unsigned int), 
                                  histogram.data(), NULL, &event2b);
            event2b.wait();
            metrics[c][1].transfer_time = (event2b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                                         event2b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            metrics[c][1].total_time = metrics[c][1].kernel_time + metrics[c][1].transfer_time;
            metrics[c][1].work = image_size;
            metrics[c][1].span = 2;

            CImg<unsigned char> hist_img(num_bins, 200, 1, 1, 0);
            const unsigned char white[] = {255};
            unsigned int max_hist = *std::max_element(histogram.begin(), histogram.end());
            for (int x = 0; x < num_bins; x++) {
                int height = (int)((histogram[x] / (float)max_hist) * 200);
                hist_img.draw_line(x, 200, x, 200 - height, white);
            }
            char hist_title[32];
            sprintf(hist_title, "Histogram Channel %d", c + 1);
            disp_hist[c] = CImgDisplay(hist_img, hist_title);

            

            // Step 3: Cumulative histogram
            cl::Event event3a, event3b;
            const char* kernel_name = (strcmp(scan_kernel_type, "bl") == 0) ? "scan_bl" : "scan_hs";
            cl::Kernel scan_kernel(program, kernel_name);
            scan_kernel.setArg(0, dev_histogram[c]);
            scan_kernel.setArg(1, num_bins);
            queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange, cl::NDRange(num_bins), 
                                     cl::NullRange, NULL, &event3a);
            event3a.wait();
            metrics[c][2].kernel_time = (event3a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                                       event3a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;

            std::vector<unsigned int> cum_histogram(num_bins);
            queue.enqueueReadBuffer(dev_histogram[c], CL_TRUE, 0, num_bins * sizeof(unsigned int), 
                                  cum_histogram.data(), NULL, &event3b);
            event3b.wait();
            metrics[c][2].transfer_time = (event3b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                                         event3b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            metrics[c][2].total_time = metrics[c][2].kernel_time + metrics[c][2].transfer_time;
            metrics[c][2].work = (strcmp(scan_kernel_type, "bl") == 0) ? (2 * num_bins - 1) : 
                                 (num_bins * (size_t)(log2((double)num_bins)));
            metrics[c][2].span = (size_t)log2((double)num_bins);

            CImg<unsigned char> cum_hist_img(num_bins, 200, 1, 1, 0);
            unsigned int max_cum_hist = cum_histogram[num_bins - 1];
            for (int x = 0; x < num_bins; x++) {
                int height = (int)((cum_histogram[x] / (float)max_cum_hist) * 200);
                cum_hist_img.draw_line(x, 200, x, 200 - height, white);
            }
            char cum_hist_title[40];
            sprintf(cum_hist_title, "Cumulative Histogram Channel %d", c + 1);
            disp_cum_hist[c] = CImgDisplay(cum_hist_img, cum_hist_title);

            // Step 4: Normalize LUT
            cl::Event event4a, event4b;
            float scale = 65535.0f / (width * height);
            cl::Kernel normalize_kernel(program, "normalize_lut");
            normalize_kernel.setArg(0, dev_histogram[c]);
            normalize_kernel.setArg(1, dev_lut[c]);
            normalize_kernel.setArg(2, scale);
            normalize_kernel.setArg(3, num_bins);
            queue.enqueueNDRangeKernel(normalize_kernel, cl::NullRange, cl::NDRange(65536), 
                                     cl::NullRange, NULL, &event4a);
            event4a.wait();
            metrics[c][3].kernel_time = (event4a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                                       event4a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;

            std::vector<unsigned short> lut(65536);
            queue.enqueueReadBuffer(dev_lut[c], CL_TRUE, 0, 65536 * sizeof(unsigned short), 
                                  lut.data(), NULL, &event4b);
            event4b.wait();
            metrics[c][3].transfer_time = (event4b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                                         event4b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            metrics[c][3].total_time = metrics[c][3].kernel_time + metrics[c][3].transfer_time;
            metrics[c][3].work = 65536;
            metrics[c][3].span = 1;

            CImg<unsigned char> norm_cum_hist_img(num_bins, 200, 1, 1, 0);
            for (int x = 0; x < num_bins; x++) {
                int lut_index = (int)((float)x / num_bins * 65536);
                int height = (int)((lut[lut_index] / 65535.0f) * 200);
                norm_cum_hist_img.draw_line(x, 200, x, 200 - height, white);
            }
            char norm_cum_hist_title[48];
            sprintf(norm_cum_hist_title, "Normalized Cumulative Histogram Channel %d", c + 1);
            disp_norm_cum_hist[c] = CImgDisplay(norm_cum_hist_img, norm_cum_hist_title);
            

            // Step 5: Back projection
            cl::Event event5a, event5b;
            cl::Kernel backproject_kernel(program, "back_project");
            backproject_kernel.setArg(0, dev_image_input[c]);
            backproject_kernel.setArg(1, dev_image_output[c]);
            backproject_kernel.setArg(2, dev_lut[c]);
            queue.enqueueNDRangeKernel(backproject_kernel, cl::NullRange, cl::NDRange(image_size), 
                                     cl::NullRange, NULL, &event5a);
            event5a.wait();
            metrics[c][4].kernel_time = (event5a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                                       event5a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;

            std::vector<unsigned short> output_buffer(image_size);
            queue.enqueueReadBuffer(dev_image_output[c], CL_TRUE, 0, image_size * sizeof(unsigned short), 
                                  output_buffer.data(), NULL, &event5b);
            event5b.wait();
            metrics[c][4].transfer_time = (event5b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - 
                                         event5b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            metrics[c][4].total_time = metrics[c][4].kernel_time + metrics[c][4].transfer_time;
            metrics[c][4].work = image_size;
            metrics[c][4].span = 1;

            // Update input channel with output
            cimg_forXY(input_channels[c], x, y) {
                input_channels[c](x, y) = output_buffer[x + y * width];
            }
        }

        // Print metrics
        double combined_total_time = 0.0;
        const char* scan_name = (strcmp(scan_kernel_type, "bl") == 0) ? "Blelloch" : "Hillis-Steele";
        for (int c = 0; c < channels; c++) {
            std::cout << "\nPerformance Metrics (seconds) and Complexity for Channel " << (c + 1) 
                      << " (Bins: " << num_bins << ", Scan Kernel: " << scan_name << "):\n";
            std::cout << "1: Input Transfer and Initialization\n";
            std::cout << "  Transfer Time: " << metrics[c][0].transfer_time << "\n";
            std::cout << "  Kernel Time: " << metrics[c][0].kernel_time << "\n";
            std::cout << "  Total Time: " << metrics[c][0].total_time << "\n";
            std::cout << "  Work: " << metrics[c][0].work << " operations\n";
            std::cout << "  Span: " << metrics[c][0].span << " steps\n";

            std::cout << "2: Histogram Calculation\n";
            std::cout << "  Transfer Time: " << metrics[c][1].transfer_time << "\n";
            std::cout << "  Kernel Time: " << metrics[c][1].kernel_time << "\n";
            std::cout << "  Total Time: " << metrics[c][1].total_time << "\n";
            std::cout << "  Work: " << metrics[c][1].work << " operations\n";
            std::cout << "  Span: " << metrics[c][1].span << " steps\n";

            std::cout << "3: Cumulative Histogram (" << scan_name << ")\n";
            std::cout << "  Transfer Time: " << metrics[c][2].transfer_time << "\n";
            std::cout << "  Kernel Time: " << metrics[c][2].kernel_time << "\n";
            std::cout << "  Total Time: " << metrics[c][2].total_time << "\n";
            std::cout << "  Work: " << metrics[c][2].work << " operations\n";
            std::cout << "  Span: " << metrics[c][2].span << " steps\n";

            std::cout << "4: Normalize LUT\n";
            std::cout << "  Transfer Time: " << metrics[c][3].transfer_time << "\n";
            std::cout << "  Kernel Time: " << metrics[c][3].kernel_time << "\n";
            std::cout << "  Total Time: " << metrics[c][3].total_time << "\n";
            std::cout << "  Work: " << metrics[c][3].work << " operations\n";
            std::cout << "  Span: " << metrics[c][3].span << " steps\n";

            std::cout << "5: Back Projection\n";
            std::cout << "  Transfer Time: " << metrics[c][4].transfer_time << "\n";
            std::cout << "  Kernel Time: " << metrics[c][4].kernel_time << "\n";
            std::cout << "  Total Time: " << metrics[c][4].total_time << "\n";
            std::cout << "  Work: " << metrics[c][4].work << " operations\n";
            std::cout << "  Span: " << metrics[c][4].span << " steps\n";

            double overall_total_time = metrics[c][0].total_time + metrics[c][1].total_time + 
                                      metrics[c][2].total_time + metrics[c][3].total_time + 
                                      metrics[c][4].total_time;
            std::cout << "Overall Total Time for Channel " << (c + 1) << ": " 
                      << overall_total_time << " seconds\n";
            combined_total_time += overall_total_time;
        }

        if (channels > 1) {
            std::cout << "\nTotal Time for ALL Channels Combined (RGB Image, Scan Kernel: " 
                      << scan_name << "): " << combined_total_time << " seconds\n";
        }

        // Combine channels into output image
        CImg<unsigned short> output_image(width, height, 1, channels);
        cimg_forXY(output_image, x, y) {
            for (int c = 0; c < channels; c++) {
                output_image(x, y, 0, c) = input_channels[c](x, y);
            }
        }
        CImgDisplay disp_output(output_image, "Equalized Image");

        // Wait for user to close windows
        bool all_closed = false;
        while (!all_closed) {
            all_closed = disp_input.is_closed() && disp_output.is_closed();
            for (int c = 0; c < channels; c++) {
                all_closed &= disp_hist[c].is_closed() && disp_cum_hist[c].is_closed() && 
                            disp_norm_cum_hist[c].is_closed();
            }
            disp_input.wait(1);
            disp_output.wait(1);
            for (int c = 0; c < channels; c++) {
                disp_hist[c].wait(1);
                disp_cum_hist[c].wait(1);
                disp_norm_cum_hist[c].wait(1);
            }
            if (disp_input.is_keyESC() || disp_output.is_keyESC()) break;
        }
    } catch (const cl::Error& err) {
        std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
        return 1;
    } catch (CImgException& err) {
        std::cerr << "ERROR: " << err.what() << std::endl;
        return 1;
    }

    return 0;
}