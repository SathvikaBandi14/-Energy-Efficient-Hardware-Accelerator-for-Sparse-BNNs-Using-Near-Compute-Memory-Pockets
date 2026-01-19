/**
 * Flattening Module: 128 channels * 3x3 spatial = 1152 bits
 */
#include <ap_int.h>
#include <hls_stream.h>
void flatten_to_1152(
    hls::stream<ap_uint<128>> &in_pool3, 
    hls::stream<ap_uint<1152>> &out_flatten
) {
    // Buffer for the 3x3 spatial grid (9 pixels total)
    ap_uint<128> pixel_buffer[9];

    // 1. Read all 9 pixels from the stream
    for(int i = 0; i < 9; i++) {
        pixel_buffer[i] = in_pool3.read();
    }

    ap_uint<1152> flat_vector = 0;

    // 2. Transpose: PyTorch expects Channel-Major order
    // Outer loop must be CHANNELS (0 to 127)
    for(int ch = 0; ch < 128; ch++) {
        // Inner loop is PIXELS (0 to 8)
        for(int p = 0; p < 9; p++) {
            // Index 0-8 is Ch0, 9-17 is Ch1, etc.
            int bit_idx = (ch * 9) + p;
            
            // Extract the 'ch'-th bit from the 'p'-th pixel
            flat_vector[bit_idx] = pixel_buffer[p][ch];
        }
    }

    out_flatten.write(flat_vector);
}