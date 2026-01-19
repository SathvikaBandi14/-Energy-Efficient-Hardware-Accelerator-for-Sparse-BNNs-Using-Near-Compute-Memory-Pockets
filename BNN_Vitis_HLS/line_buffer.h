#ifndef LINE_BUFFER_H
#define LINE_BUFFER_H

#include <ap_int.h>
#include <hls_stream.h>

void line_buffer_bin_3x3(
    hls::stream<ap_uint<1>> &pixel_in,
    hls::stream<ap_uint<9>> &window_out
);

#endif
