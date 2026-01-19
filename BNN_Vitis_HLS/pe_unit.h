#ifndef SPARSE_ENGINE_H
#define SPARSE_ENGINE_H

#include <ap_int.h>
#include <hls_stream.h>
#include "bnn_config.h"

/*
  ROWS        : number of output neurons (filters)
  COLS        : number of input bits per pixel
  NNZ         : number of nonzero weights
  NUM_PIXELS  : number of input pixels (or vectors)
  POP_W       : bitwidth for popcount (>= log2(max fan-in + 1))
  THRESH_W    : bitwidth for threshold (same as POP_W usually)
  THRESH_IN_W : bitwidth of stored threshold constants
*/

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
    const ap_uint<1> weights[NNZ],
    const unsigned short col_indices[NNZ],
    const unsigned int row_ptr[ROWS + 1],
    const ap_uint<THRESH_IN_W> thresholds_in[ROWS]
) {
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=col_indices cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=row_ptr complete
#pragma HLS ARRAY_PARTITION variable=thresholds_in complete

    // -------------------------------
    // Widen thresholds (compile-time)
    // -------------------------------
    ap_uint<THRESH_W> thresholds[ROWS];
#pragma HLS ARRAY_PARTITION variable=thresholds complete

    for (int i = 0; i < ROWS; i++) {
#pragma HLS UNROLL
        thresholds[i] = thresholds_in[i];
    }

pixel_loop:
    for (int p = 0; p < NUM_PIXELS; p++) {
#pragma HLS PIPELINE II=1

        ap_uint<COLS> input_bits = in_stream.read();
        ap_uint<ROWS> output_bits_val = 0;

    neuron_loop:
        for (int i = 0; i < ROWS; i++) {
#pragma HLS UNROLL

            ap_uint<POP_W> popcount = 0;

            int start = row_ptr[i];
            int end   = row_ptr[i + 1];

        weight_loop:
            for (int j = start; j < end; j++) {
#pragma HLS UNROLL factor=4
                popcount += (weights[j] == input_bits[col_indices[j]]);
            }

#ifndef __SYNTHESIS__

            // ================= FC2 DEBUG =================
            if (ROWS == FC2_ROWS && NUM_PIXELS == 1) {

                // Print FC2 stats
                printf(
                    "[FC2] digit=%d | popcount=%d | threshold=%d | fire=%d\n",
                    i,
                    (int)popcount,
                    (int)thresholds[i],
                    (popcount >= thresholds[i])
                );

                // TEMP: tighten FC2 threshold
                output_bits_val[i] = (popcount >= (thresholds[i] + 80));
            }
            else {
                // Normal behavior for all other layers
                output_bits_val[i] = (popcount >= thresholds[i]);
            }

#else
            // ================= SYNTHESIS =================
            output_bits_val[i] = (popcount >= thresholds[i]);
#endif
        }

        out_stream.write(output_bits_val);
    }
}

#endif
