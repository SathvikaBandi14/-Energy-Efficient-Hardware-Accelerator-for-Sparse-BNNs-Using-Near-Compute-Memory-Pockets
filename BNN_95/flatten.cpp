/**
 * Flattening Module: 128 channels * 1x1 spatial = 128 bits
 * Since spatial is 1x1, no transposition or buffering is needed.
 */
#include <ap_int.h>
#include <hls_stream.h>
void flatten(
    hls::stream<ap_uint<128>> &in_pool2, 
    hls::stream<ap_uint<6272>> &out_flatten
) {
    // This buffer physically assembles the 7x7 grid
    ap_uint<6272> wide_vector = 0;

    // Iterate through all 49 spatial positions (7x7)
    flatten_loop: for (int i = 0; i < 49; i++) {
        #pragma HLS PIPELINE II=1
        
        // Read the 128 channels for the current pixel
        ap_uint<128> pixel_channels = in_pool2.read();
        
        // Map the 128 bits into the correct "slot" in the 6272-bit vector
        // Slots: 0-127, 128-255, ..., 6144-6271
        wide_vector.range((i + 1) * 128 - 1, i * 128) = pixel_channels;
    }

    // Now send the complete "flattened" image to FC1
    out_flatten.write(wide_vector);
}