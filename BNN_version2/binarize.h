#ifndef BINARIZE_H
#define BINARIZE_H

#include <ap_int.h>
#include <hls_stream.h>

void binarize(
    hls::stream<ap_uint<8>> &in_stream,
    hls::stream<ap_uint<1>> &out_stream
);

#endif
