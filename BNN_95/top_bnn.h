#ifndef TOP_BNN_H
#define TOP_BNN_H
#include <ap_int.h>
#include <hls_stream.h>

// --- Top Level Accelerator ---
// Input: 8-bit grayscale pixels
// Output: 10-bit integer (representing digits 0-9)
void bnn_accelerator(
    hls::stream<ap_uint<8>>  &in_pixels,
    hls::stream<ap_uint<10>> &output_class
);

// --- Hardcoded Layer Prototypes ---

// CONV1: 1 channel in -> 64 channels out
void compute_conv1(
    hls::stream<ap_uint<8>> &in, 
    hls::stream<ap_uint<64>> &out
);

// CONV2: 64 channels in -> 128 channels out
void compute_conv2(
    hls::stream<ap_uint<64>> &in, 
    hls::stream<ap_uint<128>> &out
);

// FLATTEN: 128 channels * 7 * 7 spatial -> 6272-bit vector
void flatten(
    hls::stream<ap_uint<128>> &in, 
    hls::stream<ap_uint<6272>> &out
);

// FC1: 6272 bits in -> 512 bits out
void compute_fc1(
    hls::stream<ap_uint<6272>> &in, 
    hls::stream<ap_uint<512>> &out
);

// FC2: 512 bits in -> Final 10-class decision
void compute_fc2(
    hls::stream<ap_uint<512>> &in, 
    hls::stream<ap_uint<10>> &out
);

#endif