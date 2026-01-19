#ifndef TOP_BNN_H
#define TOP_BNN_H

#include <ap_int.h>
#include <hls_stream.h>
#include "bnn_config.h"

// ============================================================
// TOP-LEVEL BNN ACCELERATOR
// ============================================================
void bnn_accelerator(
    hls::stream<ap_uint<8>> &in_pixels,
    hls::stream<ap_uint<FC2_ROWS>> &output_class
);

// ============================================================
// CONVOLUTION LAYERS (PURE BNN)
// ============================================================

// Conv1: 28x28, 3x3, Cin=1
void compute_conv1(
    hls::stream<ap_uint<8>> &in_pixels,
    hls::stream<ap_uint<CONV1_ROWS>> &out_bits
);

// Conv2: 14x14, 3x3, Cin=32
void compute_conv2(
    hls::stream<ap_uint<CONV2_CH>> &in_pixels,
    hls::stream<ap_uint<CONV2_ROWS>> &out_bits
);

// Conv3: 7x7, 3x3, Cin=64
void compute_conv3(
    hls::stream<ap_uint<CONV3_CH>> &in_pixels,
    hls::stream<ap_uint<CONV3_ROWS>> &out_bits
);

// ============================================================
// FULLY CONNECTED LAYERS
// ============================================================

void compute_fc1(
    hls::stream<ap_uint<FC1_COLS>> &in_vector,
    hls::stream<ap_uint<FC1_ROWS>> &out_bits
);

void compute_fc2(
    hls::stream<ap_uint<FC2_COLS>> &in_vector,
    hls::stream<ap_uint<FC2_ROWS>> &out_bits
);

// ============================================================
// FLATTEN
// ============================================================
void flatten_to_1152(
    hls::stream<ap_uint<CONV3_ROWS>> &pool3_out,
    hls::stream<ap_uint<FC1_COLS>> &fc1_in
);

#endif
