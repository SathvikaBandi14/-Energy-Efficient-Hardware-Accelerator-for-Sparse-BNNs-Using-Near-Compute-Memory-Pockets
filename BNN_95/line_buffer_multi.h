#ifndef LINE_BUFFER_MULTI_H
#define LINE_BUFFER_MULTI_H

#include <ap_int.h>
#include <hls_stream.h>
template<int CH, int DIM>
void line_buffer_multi(
    hls::stream<ap_uint<CH>> &in_stream,
    hls::stream<ap_uint<CH * 9>> &patch_out
) {
    static ap_uint<CH> line_buf[2][DIM];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    static ap_uint<CH> window[3][3];
    #pragma HLS ARRAY_PARTITION variable=window complete

    // Loop from 0 to DIM+1 (e.g., 16 for a 14x14) to handle padding edges
    Row_Loop: for (int r = 0; r < DIM + 2; r++) {
        Col_Loop: for (int c = 0; c < DIM + 2; c++) {
            #pragma HLS PIPELINE II=1
            
            ap_uint<CH> current_pixel = 0;
            // Only read from stream if we are inside the actual image (not the padding)
            if (r >= 1 && r <= DIM && c >= 1 && c <= DIM) {
                current_pixel = in_stream.read();
            }

            // Shift Window Horizontal
            for (int i = 0; i < 3; i++) {
                #pragma HLS UNROLL
                window[i][0] = window[i][1];
                window[i][1] = window[i][2];
            }

            // Fetch from line buffer and Update Window
            ap_uint<CH> lbuf0 = (c < DIM) ? line_buf[0][c] : ap_uint<CH>(0);
            ap_uint<CH> lbuf1 = (c < DIM) ? line_buf[1][c] : ap_uint<CH>(0);

            window[0][2] = lbuf0;
            window[1][2] = lbuf1;
            window[2][2] = current_pixel;

            // Update Line Buffer (only within image columns)
            if (c < DIM) {
                line_buf[0][c] = lbuf1;
                line_buf[1][c] = current_pixel;
            }

            // Output Valid 3x3 Window (now results in same DIM as input)
            if (r >= 2 && c >= 2) {
                ap_uint<CH * 9> flat_patch;
                for(int ch = 0; ch < CH; ch++) {
                    #pragma HLS UNROLL
                    for(int wi = 0; wi < 3; wi++) {
                        for(int wj = 0; wj < 3; wj++) {
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