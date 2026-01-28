#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <ap_int.h>
#include <hls_stream.h>

/**
 * Binary 2x2 MaxPool (stride = 2)
 * Logic: BNN MaxPool is a bitwise OR of the 2x2 window.
 */
template<int CH, int IN_DIM>
void binary_maxpool_2x2(
    hls::stream<ap_uint<CH>> &in_stream,
    hls::stream<ap_uint<CH>> &out_stream
) {
    // Buffer for the even row to be ORed with the following odd row
    ap_uint<CH> row_buf[IN_DIM];
    #pragma HLS ARRAY_PARTITION variable=row_buf cyclic factor=2
    
    ap_uint<CH> prev_col_val; 

    for (int r = 0; r < IN_DIM; r++) {
        for (int c = 0; c < IN_DIM; c++) {
            #pragma HLS PIPELINE II=1
            
            ap_uint<CH> curr = in_stream.read();

            if ((r % 2) == 0) {
                // Row 0, 2, 4... Store for later
                row_buf[c] = curr;
            } else {
                // Row 1, 3, 5... Perform vertical OR
                if ((c % 2) == 0) {
                    // Col 0, 2, 4... Store for horizontal OR
                    prev_col_val = curr | row_buf[c];
                } else {
                    // Col 1, 3, 5... We have the full 2x2 block
                    // OR the current result with the previous buffered vertical pair
                    ap_uint<CH> pooled = prev_col_val | (curr | row_buf[c]);
                    out_stream.write(pooled);
                }
            }
        }
    }
}
/**
 * Global Count-Based Pool (k-of-N)
 * Reduces IN_DIM x IN_DIM spatial map to 1x1
 * A channel survives only if it fires >= K times
 */
template<int CH, int IN_DIM, int K>
void global_maxpool(
    hls::stream<ap_uint<CH>> &in_stream,
    hls::stream<ap_uint<CH>> &out_stream
) {
    // Counter per channel
    ap_uint<4> count[CH];
    #pragma HLS ARRAY_PARTITION variable=count complete

    // Initialize counters
    for (int b = 0; b < CH; b++) {
        #pragma HLS UNROLL
        count[b] = 0;
    }

    // Accumulate counts across spatial positions
    for (int i = 0; i < IN_DIM * IN_DIM; i++) {
        #pragma HLS PIPELINE II=1
        ap_uint<CH> val = in_stream.read();

        for (int b = 0; b < CH; b++) {
            #pragma HLS UNROLL
            count[b] += val[b];
        }
    }

    // Threshold counts to produce pooled output
    ap_uint<CH> acc = 0;
    for (int b = 0; b < CH; b++) {
        #pragma HLS UNROLL
        acc[b] = (count[b] >= K);
    }

#ifndef __SYNTHESIS__
    int set_bits = 0;
    for (int b = 0; b < CH; b++) {
        if (acc[b] == 1) set_bits++;
    }
    std::cout << "Final Vector Density (K=" << K
              << "): " << set_bits << " / " << CH << std::endl;
#endif

    out_stream.write(acc);
}
#endif