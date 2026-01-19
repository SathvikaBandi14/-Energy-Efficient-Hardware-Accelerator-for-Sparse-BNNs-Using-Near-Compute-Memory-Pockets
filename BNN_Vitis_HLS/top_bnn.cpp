#include "top_bnn.h"
#include <ap_int.h>
#include <hls_stream.h>
#include "maxpool.h"

// ============================================================
// TOP-LEVEL BNN ACCELERATOR (PURE INTEGER BNN)
// ============================================================
void bnn_accelerator(
    hls::stream<ap_uint<8>> &in_pixels,
    hls::stream<ap_uint<FC2_ROWS>> &output_class
) {
#pragma HLS DATAFLOW

    // ================= STREAMS =================
    hls::stream<ap_uint<CONV1_ROWS>> conv1_out("conv1");
    hls::stream<ap_uint<CONV2_CH>>   pool1_out("pool1");

    hls::stream<ap_uint<CONV2_ROWS>> conv2_out("conv2");
    hls::stream<ap_uint<CONV3_CH>>   pool2_out("pool2");

    hls::stream<ap_uint<CONV3_ROWS>> conv3_out("conv3");
    hls::stream<ap_uint<CONV3_ROWS>> pool3_out("pool3");

    hls::stream<ap_uint<FC1_COLS>>   flatten_out("flatten");
    hls::stream<ap_uint<FC1_ROWS>>   fc1_out("fc1");

    // ================= STREAM DEPTHS =================
#pragma HLS STREAM variable=conv1_out   depth=784
#pragma HLS STREAM variable=pool1_out   depth=196

#pragma HLS STREAM variable=conv2_out   depth=196
#pragma HLS STREAM variable=pool2_out   depth=49

#pragma HLS STREAM variable=conv3_out   depth=49
#pragma HLS STREAM variable=pool3_out   depth=16

#pragma HLS STREAM variable=flatten_out depth=2
#pragma HLS STREAM variable=fc1_out     depth=2

    // ================= PIPELINE =================
    compute_conv1(in_pixels, conv1_out);
    binary_maxpool_2x2<CONV1_ROWS, 28>(conv1_out, pool1_out);

    compute_conv2(pool1_out, conv2_out);
    binary_maxpool_2x2<CONV2_ROWS, 14>(conv2_out, pool2_out);

    compute_conv3(pool2_out, conv3_out);
    binary_maxpool_2x2<CONV3_ROWS, 7>(conv3_out, pool3_out);

    flatten_to_1152(pool3_out, flatten_out);

    compute_fc1(flatten_out, fc1_out);
    compute_fc2(fc1_out, output_class);
}
