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
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 69, 192, 254, 253, 200, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 187, 252, 252, 253, 252, 252, 237, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 169, 252, 252, 235, 144, 247, 252, 247, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 218, 252, 247, 162, 14, 29, 232, 252, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 173, 47, 0, 45, 219, 252, 190, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 114, 236, 255, 239, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180, 252, 252, 186, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 87, 251, 238, 99, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 252, 170, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 236, 252, 235, 194, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 176, 211, 237, 253, 147, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99, 231, 252, 237, 146, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 179, 245, 252, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 242, 222, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 167, 253, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 69, 236, 230, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 7, 0, 36, 86, 164, 195, 252, 252, 131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 215, 233, 239, 234, 232, 241, 253, 252, 252, 238, 187, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 242, 253, 252, 252, 252, 252, 253, 252, 212, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 165, 217, 147, 147, 147, 147, 112, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

int main() {
    hls::stream<ap_uint<8>> sim_input("sim_input");
    hls::stream<ap_uint<FC2_ROWS>> sim_output("sim_output");  

    std::cout << "--- Starting BNN Simulation ---" << std::endl;

    // 1. Visualize the input array
    std::cout << "Input Image (Binarized Preview):" << std::endl;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (mnist_sample[i * 28 + j] >= 33) std::cout << "#";
            else std::cout << " ";
        }
        std::cout << "|" << std::endl;
    }

    // 2. Feed ACTUAL MNIST data into the accelerator stream
    std::cout << "--- Feeding MNIST Image Data ---" << std::endl;
    for (int i = 0; i < 784; i++) {
        // Now writing the actual values from the mnist_sample array
        sim_input.write((ap_uint<8>)mnist_sample[i]);
    }

    // 3. Execute the accelerator
    bnn_accelerator(sim_input, sim_output);

    // 4. Capture and display results
    if (!sim_output.empty()) {
        ap_uint<FC2_ROWS> result = sim_output.read();
        
        std::cout << "Raw Hardware Output: 0x" << std::hex << (unsigned int)result << std::dec << std::endl;
        
        int predicted_class = -1;
        int bits_count = 0;
        for(int i = 0; i < 10; i++) {
            if(result[i]) {
                if(predicted_class == -1) predicted_class = i;
                bits_count++;
            }
        }

        if (bits_count == 0) {
            std::cout << "Final Result: No digits detected (all bits low)." << std::endl;
        } else if (bits_count > 1) {
            std::cout << "Final Result: Multiple digits detected (Check thresholds!)." << std::endl;
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