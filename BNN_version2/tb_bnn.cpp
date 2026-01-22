#include <iostream>
#include <hls_stream.h>
#include <ap_int.h>
#include "top_bnn.h"
#include "bnn_config.h"

// Input changed from ap_uint<1> to ap_uint<8>
void bnn_accelerator(
    hls::stream<ap_uint<8>> &in_pixels, 
    hls::stream<ap_uint<FC2_ROWS>> &output_class 
);

// Your MNIST data array
static const unsigned char mnist_sample[784] = {
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 106, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 98, 157, 252, 239, 209, 87, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71, 216, 254, 254, 254, 254, 254, 250, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 191, 251, 199, 139, 61, 61, 173, 255, 141, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71, 191, 254, 241, 0, 0, 0, 0, 16, 175, 254, 215, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 207, 254, 226, 97, 0, 0, 0, 0, 0, 12, 189, 254, 213, 103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 188, 168, 254, 82, 0, 0, 0, 0, 0, 0, 0, 74, 254, 254, 214, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 235, 254, 230, 47, 0, 0, 0, 0, 0, 0, 0, 1, 74, 249, 254, 253, 133, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 106, 252, 228, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 193, 254, 254, 213, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 254, 178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 190, 254, 247, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 193, 254, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 254, 254, 197, 0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 254, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 249, 254, 236, 13, 0, 0, 0, 0, 0, 0, 0, 0, 118, 254, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 197, 254, 254, 121, 0, 0, 0, 0, 0, 0, 0, 0, 118, 254, 178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 254, 254, 187, 0, 0, 0, 0, 0, 0, 0, 0, 6, 236, 178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 254, 246, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 235, 203, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 200, 254, 240, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 103, 211, 109, 63, 63, 63, 63, 29, 63, 29, 63, 63, 63, 194, 218, 254, 254, 238, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 105, 241, 254, 254, 254, 214, 254, 214, 254, 254, 254, 254, 254, 244, 116, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65, 178, 234, 234, 185, 208, 234, 234, 234, 166, 96, 96, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

int main() {
    hls::stream<ap_uint<8>>  sim_input("sim_input");
    hls::stream<ap_uint<10>> sim_output("sim_output");  // FIX

    std::cout << "--- Starting BNN Simulation ---" << std::endl;

    // Visualize input
    std::cout << "Input Image (Binarized Preview):" << std::endl;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (mnist_sample[i * 28 + j] >= 33) std::cout << "#";
            else std::cout << " ";
        }
        std::cout << "|" << std::endl;
    }

    // Feed MNIST pixels
    for (int i = 0; i < 784; i++) {
        sim_input.write((ap_uint<8>)mnist_sample[i]);
    }

    // Run accelerator
    bnn_accelerator(sim_input, sim_output);

    // Read output
    if (!sim_output.empty()) {
        ap_uint<10> result = sim_output.read();   // FIX

        std::cout << "Raw Hardware Output: 0x"
                  << std::hex << (unsigned int)result << std::dec << std::endl;

        int predicted_class = -1;
        int bits_count = 0;

        for (int i = 0; i < 10; i++) {
            if (result[i]) {
                if (predicted_class == -1) predicted_class = i;
                bits_count++;
            }
        }

        if (bits_count == 0) {
            std::cout << "Final Result: No digits detected." << std::endl;
        } else if (bits_count > 1) {
            std::cout << "Final Result: Multiple digits detected." << std::endl;
            std::cout << "First detected digit: " << predicted_class << std::endl;
        } else {
            std::cout << "Final Result: SUCCESS" << std::endl;
            std::cout << "Predicted Digit: " << predicted_class << std::endl;
        }
    } else {
        std::cout << "Error: No data in output stream." << std::endl;
    }

    return 0;
}
