#ifndef PE_UNIT_H
#define PE_UNIT_H

#include <ap_int.h>
#include <hls_stream.h>

template<
    int ROWS,
    int COLS,
    int NNZ,
    int NUM_PIXELS,
    int POP_W,
    int THRESH_W,
    int THRESH_IN_W
>
void sparse_bnn_layer(
    hls::stream<ap_uint<COLS>> &in_stream,
    hls::stream<ap_uint<ROWS>> &out_stream,
    const unsigned int col_indices[NNZ],
    const unsigned int row_ptr[ROWS + 1],
    const ap_int<THRESH_IN_W> thresholds_in[ROWS]
) {
#pragma HLS INLINE off

pixel_loop:
    for (int p = 0; p < NUM_PIXELS; p++) {
        #pragma HLS PIPELINE II=1

        ap_uint<COLS> input_bits = in_stream.read();
        ap_uint<ROWS> output_bits_val = 0;

    neuron_loop:
        for (int i = 0; i < ROWS; i++) {
            #pragma HLS UNROLL

            ap_uint<POP_W> popcount = 0;
            unsigned int start = row_ptr[i];
            unsigned int end   = row_ptr[i + 1];

        weight_loop:
            for (unsigned int j = start; j < end; j++) {
                // Sparse +1-only logic
                if (input_bits[col_indices[j]]) {
                    popcount++;
                }
            }

            // ✔ Correct thresholding (popcount domain)
            output_bits_val[i] = (popcount >= thresholds_in[i]);
        }

        out_stream.write(output_bits_val);
    }
}

#endif
