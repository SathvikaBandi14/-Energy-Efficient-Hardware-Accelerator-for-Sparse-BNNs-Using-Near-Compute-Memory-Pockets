#include "sparse_params.h"

/**
 * input_quantizer: Transforms 8-bit pixels into 1-bit activations.
 * * @param raw_pixels: Array of 784 pixels (28x28 MNIST image) in 8-bit format.
 * @param quantized_bits: The output 1-bit activation vector used by the PE Engine.
 */
void input_quantizer(
    ap_uint<8> raw_pixels[784], 
    bit quantized_bits[784]
) {
    // We use a pipeline directive with an Initiation Interval (II) of 1.
    // This means the hardware can process one pixel per clock cycle.
    QUANTIZE_LOOP: for(int i = 0; i < 784; i++) {
        #pragma HLS PIPELINE II=1
        
        // Point (b): Binarization Logic
        // pixels are compared against a fixed threshold (128).
        // 0-127 becomes '0'(-1 in BNN math)
        // 128-255 becomes '1'(+1 in BNN math)
        if (raw_pixels[i] >= 128) {
            quantized_bits[i] = 1;
        } else {
            quantized_bits[i] = 0;
        }
    }
}