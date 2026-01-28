#include <iostream>
#include <hls_stream.h>
#include <ap_int.h>
#include "top_bnn.h"
#include "bnn_config.h"

// Input changed from ap_uint<1> to ap_uint<8>
void bnn_accelerator(
    hls::stream<ap_uint<8>> &in_pixels, 
    hls::stream<ap_uint<10>> &output_class 
);

// Your MNIST data array
static const unsigned char mnist_sample[784] = {
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 116, 125, 171, 255, 255, 150, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 253, 253, 253, 253, 253, 253, 218, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 169, 253, 253, 253, 213, 142, 176, 253, 253, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 250, 253, 210, 32, 12, 0, 6, 206, 253, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 251, 210, 25, 0, 0, 0, 122, 248, 253, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 18, 0, 0, 0, 0, 209, 253, 253, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 117, 247, 253, 198, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 247, 253, 231, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 253, 253, 144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 246, 253, 159, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 234, 253, 233, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 198, 253, 253, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 248, 253, 189, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 200, 253, 253, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 253, 253, 173, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 43, 20, 20, 20, 20, 5, 0, 5, 20, 20, 37, 150, 150, 150, 147, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 253, 253, 253, 253, 253, 168, 143, 166, 253, 253, 253, 253, 253, 253, 253, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 249, 247, 247, 169, 117, 117, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 123, 123, 123, 166, 253, 253, 253, 155, 123, 123, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
int main() {
    // Match the hardcoded types in top_bnn.h
    hls::stream<ap_uint<8>>  sim_input("sim_input");
    hls::stream<ap_uint<10>> sim_output("sim_output"); 

    std::cout << "--- Starting BNN Hardware Simulation ---" << std::endl;

    // 1. Visualize Input Pattern
    std::cout << "Input Image Preview:" << std::endl;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            // Simple binarization for console visualization (threshold 33)
            if (mnist_sample[i * 28 + j] >= 33) std::cout << "#";
            else std::cout << " ";
        }
        std::cout << "|" << std::endl;
    }

    // 2. Feed Pixels to Accelerator
    for (int i = 0; i < 784; i++) {
        sim_input.write((ap_uint<8>)mnist_sample[i]);
    }

    // 3. Execute Top Level Function
    // This calls your Dataflow pipeline: Conv1 -> Pool1 -> Conv2 -> Pool2 -> Flatten -> FC1 -> FC2
    bnn_accelerator(sim_input, sim_output);

    // 4. Handle Result
    if (!sim_output.empty()) {
        // Since FC2 writes the winning index (0-9), read it as a single value
        ap_uint<10> predicted_digit = sim_output.read(); 

        std::cout << "------------------------------------" << std::endl;
        std::cout << "Hardware Prediction: " << (unsigned int)predicted_digit << std::endl;
        std::cout << "------------------------------------" << std::endl;

        // Verification logic
        if (predicted_digit >= 0 && predicted_digit <= 9) {
            std::cout << "Simulation Result: VALID DIGIT DETECTED" << std::endl;
        } else {
            std::cout << "Simulation Result: ERROR (Out of Range)" << std::endl;
        }
    } else {
        std::cout << "Simulation Error: Output stream is empty. Check your stream depths and padding logic." << std::endl;
    }

    return 0;
}
