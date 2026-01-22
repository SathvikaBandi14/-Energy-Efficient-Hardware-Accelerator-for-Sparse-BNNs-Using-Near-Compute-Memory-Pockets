#ifndef LINE_BUFFER_MULTI_H
#define LINE_BUFFER_MULTI_H

#include <ap_int.h>
#include <hls_stream.h>

template<int CH, int DIM>
void line_buffer_multi_valid(
    hls::stream<ap_uint<CH>> &in_stream,
    hls::stream<ap_uint<CH * 9>> &patch_out
) {
    // line_buf stores the previous two rows
    static ap_uint<CH> line_buf[2][DIM];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    
    // window stores the current 3x3 patch being processed
    static ap_uint<CH> window[3][3];
    #pragma HLS ARRAY_PARTITION variable=window complete

    Row_Loop: for (int r = 0; r < DIM; r++) {
        Col_Loop: for (int c = 0; c < DIM; c++) {
            #pragma HLS PIPELINE II=1
            
            // 1. Shift Window Horizontal (Shift left to make room for new column)
            for (int i = 0; i < 3; i++) {
                #pragma HLS UNROLL
                window[i][0] = window[i][1];
                window[i][1] = window[i][2];
            }

            // 2. Fetch data
            ap_uint<CH> current_pixel = in_stream.read();
            ap_uint<CH> lbuf0_val = line_buf[0][c];
            ap_uint<CH> lbuf1_val = line_buf[1][c];

            // 3. Update Window with new column data
            window[0][2] = lbuf0_val;
            window[1][2] = lbuf1_val;
            window[2][2] = current_pixel;

            // 4. Update Line Buffer for the next row
            line_buf[0][c] = lbuf1_val;
            line_buf[1][c] = current_pixel;

            // 5. Output Packing (Only when window is full)
            // For a 3x3 kernel, we need at least 3 rows (r>=2) and 3 cols (c>=2)
            if (r >= 2 && c >= 2) {
                ap_uint<CH * 9> flat_patch;
                
                // Pack exactly as PyTorch CHW expect
                for(int ch = 0; ch < CH; ch++) {
                    #pragma HLS UNROLL
                    for(int wi = 0; wi < 3; wi++) {
                        for(int wj = 0; wj < 3; wj++) {
                            // Bit offset logic: (Channel * spatial_size) + spatial_offset
                            int offset = (ch * 9) + (wi * 3 + wj);
                            flat_patch[offset] = window[wi][wj][ch];
                        }
                    }
                }
                patch_out.write(flat_patch);
            }
        }
    }
}
#endif