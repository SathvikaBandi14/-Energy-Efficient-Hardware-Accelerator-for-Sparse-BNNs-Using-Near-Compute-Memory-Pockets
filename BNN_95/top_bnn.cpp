#include "top_bnn.h"
#include "conv1_data.h"
#include "conv2_data.h"
#include "conv3_data.h"
#include "fc1_data.h"
#include "fc2_data.h"
void bnn_accelerator(
    hls::stream<ap_uint<8>>  &in_pixels,
    hls::stream<ap_uint<10>> &output_class
) {
    #pragma HLS DATAFLOW 

    // Internal streams matching your 95% accuracy model dimensions
    hls::stream<ap_uint<64>>   pool1_out("pool1");   // Conv1 -> Pool1
    hls::stream<ap_uint<128>>  pool2_out("pool2");   // Conv2 -> Pool2
    hls::stream<ap_uint<6272>> flatten_out("flatten"); // The "view(-1)" result
    hls::stream<ap_uint<512>>  fc1_out("fc1");

    // Depth settings for Padding=1 flow (28x28 and 14x14)
    #pragma HLS STREAM variable=pool1_out   depth=784 
    #pragma HLS STREAM variable=pool2_out   depth=196
    #pragma HLS STREAM variable=flatten_out depth=1
    #pragma HLS STREAM variable=fc1_out     depth=1

    // 1. Convolutional Stages
    compute_conv1(in_pixels, pool1_out);
    compute_conv2(pool1_out, pool2_out);

    // 2. Physical Flattening Stage (from your flatten.cpp)
    // Converts 49 cycles of 128-bit pixels into 1 cycle of 6272-bit vector
    flatten(pool2_out, flatten_out);

    // 3. Fully Connected Stages
    compute_fc1(flatten_out, fc1_out);
    compute_fc2(fc1_out, output_class);
}