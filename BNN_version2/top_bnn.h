#ifndef TOP_BNN_H
#define TOP_BNN_H

#include <ap_int.h>
#include <hls_stream.h>
#include "bnn_config.h"

// --- Top Level Accelerator ---
void bnn_accelerator(
    hls::stream<ap_uint<8>>  &in_pixels,
    hls::stream<ap_uint<10>> &output_class
);

// --- Layer Prototypes (Fixes the 'Undeclared' Error) ---
void compute_conv1(hls::stream<ap_uint<8>> &in, hls::stream<ap_uint<CONV1_ROWS>> &out);
void compute_conv2(hls::stream<ap_uint<CONV2_CH>> &in, hls::stream<ap_uint<CONV2_ROWS>> &out);
void compute_conv3(hls::stream<ap_uint<CONV3_CH>> &in, hls::stream<ap_uint<CONV3_ROWS>> &out);
void flatten_to_128(hls::stream<ap_uint<CONV3_ROWS>> &in, hls::stream<ap_uint<FC1_COLS>> &out);
void compute_fc1(hls::stream<ap_uint<FC1_COLS>> &in, hls::stream<ap_uint<FC1_ROWS>> &out);
void compute_fc2(hls::stream<ap_uint<FC1_ROWS>> &in, hls::stream<ap_uint<10>> &out);

#endif