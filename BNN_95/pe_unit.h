#ifndef PE_UNIT_H
#define PE_UNIT_H

#include <ap_int.h>
#include <hls_stream.h>

template<
    int ROWS,         // e.g., 64 for Conv1
    int COLS,         // Input size (e.g., 784 for MNIST)
    int NNZ,          // Total non-zero indices (e.g., 531 from your export)
    int NUM_PIXELS,   // Image count in batch
    int POP_W,        // Width for accumulator (ap_int<10> is safe)
    int THRESH_IN_W   // Width of exported thresholds (ap_int<10>)
>
void sparse_bnn_layer(
    hls::stream<ap_uint<COLS>> &in_stream,
    hls::stream<ap_uint<ROWS>> &out_stream,
    const unsigned int col_indices[NNZ],
    const unsigned int row_ptr[ROWS + 1],
    const ap_int<THRESH_IN_W> thresholds_in[ROWS]
) {
#pragma HLS INLINE off

// Memory Pockets: Partition arrays to allow parallel access by neurons
#pragma HLS ARRAY_PARTITION variable=row_ptr complete
#pragma HLS ARRAY_PARTITION variable=thresholds_in complete

pixel_loop:
    for (int p = 0; p < NUM_PIXELS; p++) {
        #pragma HLS PIPELINE II=1

        ap_uint<COLS> input_bits = in_stream.read();
        ap_uint<ROWS> output_bits_val = 0;

    neuron_loop:
        for (int i = 0; i < ROWS; i++) {
            // UNROLL here allows the FPGA to build 64 parallel PEs
            #pragma HLS UNROLL 

            // Accumulator must handle negative ranges for +1/-1 logic
            ap_int<POP_W> accumulator = 0; 
            unsigned int start = row_ptr[i];
            unsigned int end   = row_ptr[i + 1];

        weight_loop:
            for (unsigned int j = start; j < end; j++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=20 // Helps HLS estimation
                
                // BNN XNOR-equivalent logic: 
                // If input bit is 1, it matches the +1 weight (+1)
                // If input bit is 0, it mismatches (-1)
                if (input_bits[col_indices[j]]) {
                    accumulator++;
                } else {
                    accumulator--;
                }
            }

            // Thresholding logic: Apply the BN-folded thresholds (includes Alpha)
            // If the sum of XNORs meets the learned threshold, output 1 (binary +1)
            output_bits_val[i] = (accumulator >= thresholds_in[i]) ? 1 : 0;
        }

        out_stream.write(output_bits_val);
    }
}

#endif