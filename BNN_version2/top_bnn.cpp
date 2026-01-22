#include "top_bnn.h"
#include "conv1_data.h"
#include "conv2_data.h"
#include "conv3_data.h"
#include "fc1_data.h"
#include "fc2_data.h"

// ... keep all your compute_conv1, compute_conv2, etc. implementations here ...

// THE TOP LEVEL CALLER
void bnn_accelerator(
    hls::stream<ap_uint<8>>  &in_pixels,
    hls::stream<ap_uint<10>> &output_class
) {
    #pragma HLS DATAFLOW

    hls::stream<ap_uint<CONV1_ROWS>> pool1_out("pool1");
    hls::stream<ap_uint<CONV2_ROWS>> pool2_out("pool2");
    hls::stream<ap_uint<CONV3_ROWS>> pool3_out("pool3");
    hls::stream<ap_uint<FC1_COLS>> flatten_out("flatten");
    hls::stream<ap_uint<FC1_ROWS>> fc1_out("fc1");

    #pragma HLS STREAM variable=pool1_out  depth=169
    #pragma HLS STREAM variable=pool2_out  depth=25
    #pragma HLS STREAM variable=pool3_out  depth=1
    #pragma HLS STREAM variable=flatten_out depth=1
    #pragma HLS STREAM variable=fc1_out     depth=1

    compute_conv1(in_pixels, pool1_out);
    compute_conv2(pool1_out, pool2_out);
    compute_conv3(pool2_out, pool3_out);
    flatten_to_128(pool3_out, flatten_out);
    compute_fc1(flatten_out, fc1_out);
    compute_fc2(fc1_out, output_class);
}