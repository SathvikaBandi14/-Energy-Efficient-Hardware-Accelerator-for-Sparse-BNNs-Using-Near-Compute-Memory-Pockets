#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <ap_int.h>
#include <hls_stream.h>

/**
 * Binary 2x2 MaxPool (stride = 2)
 * Logic: output bit = OR of 2x2 window
 *
 * CH     : number of channels
 * IN_DIM : input spatial dimension (e.g. 28, 14, 7)
 *
 * Input stream  : CH-bit pixel stream
 * Output stream : CH-bit pooled pixel stream
 */
template<int CH, int IN_DIM>
void binary_maxpool_2x2(
    hls::stream<ap_uint<CH>> &in_stream,
    hls::stream<ap_uint<CH>> &out_stream
) {
    // Buffer for the previous row
    ap_uint<CH> row_buf[IN_DIM];
    // Buffer for the previous column in the current row
    ap_uint<CH> prev_col_val; 

    for (int r = 0; r < IN_DIM; r++) {
        for (int c = 0; c < IN_DIM; c++) {
            #pragma HLS PIPELINE II=1
            
            // 1. Always read to keep the stream moving
            ap_uint<CH> curr = in_stream.read();

            if ((r & 1) == 0) {
                // Even row (0, 2, 4, 6): Just buffer the data for the next row
                row_buf[c] = curr;
            } else {
                // Odd row (1, 3, 5): 
                if ((c & 1) == 0) {
                    // Even column: Buffer the left-side pixel of the 2x2
                    prev_col_val = curr;
                } else {
                    // Odd column: We have all 4 pixels of the 2x2 window
                    // row_buf[c-1] = Top-Left
                    // row_buf[c]   = Top-Right
                    // prev_col_val = Bottom-Left
                    // curr         = Bottom-Right
                    ap_uint<CH> pooled = row_buf[c-1] | row_buf[c] | prev_col_val | curr;
                    out_stream.write(pooled);
                }
            }
        }
    }
}
#endif