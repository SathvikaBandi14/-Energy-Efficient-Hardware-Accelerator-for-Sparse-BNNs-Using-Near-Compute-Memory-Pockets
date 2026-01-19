#ifndef LINE_BUFFER_MULTI_H
#define LINE_BUFFER_MULTI_H

#include <ap_int.h>
#include <hls_stream.h>

template<int CH, int DIM>
void line_buffer_multi_padded(
    hls::stream<ap_uint<CH>> &in_stream,
    hls::stream<ap_uint<CH * 9>> &patch_out
) {
    static ap_uint<CH> line_buf[2][DIM];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    static ap_uint<CH> window[3][3];
    #pragma HLS ARRAY_PARTITION variable=window complete

    static int global_pixel_cnt = 0;

    // --- RESET AT START OF IMAGE ---
    if (global_pixel_cnt == 0) {
        for (int j = 0; j < DIM; j++) {
            #pragma HLS UNROLL
            line_buf[0][j] = 0;
            line_buf[1][j] = 0;
        }
        for (int i = 0; i < 3; i++) {
            #pragma HLS UNROLL
            for (int j = 0; j < 3; j++) {
                window[i][j] = 0;
            }
        }
    }

    Row_Loop: for (int r = 0; r < DIM + 2; r++) {
        Col_Loop: for (int c = 0; c < DIM + 2; c++) {
            #pragma HLS PIPELINE II=1
            
            ap_uint<CH> current_pixel = 0;
            ap_uint<CH> lbuf0_val = 0;
            ap_uint<CH> lbuf1_val = 0;

            // 1. Shift Window Horizontal
            for (int i = 0; i < 3; i++) {
                window[i][0] = window[i][1];
                window[i][1] = window[i][2];
            }

            // 2. Fetch from Line Buffer
            if (c < DIM) {
                lbuf0_val = line_buf[0][c];
                lbuf1_val = line_buf[1][c];
            }

            // 3. Read Stream (Inside 14x14) or Padding (Border)
            if (r < DIM && c < DIM) {
                current_pixel = in_stream.read();
            } else {
                current_pixel = 0; 
            }

            // 4. Update Window
            window[0][2] = lbuf0_val;
            window[1][2] = lbuf1_val;
            window[2][2] = current_pixel;

            // 5. Update Line Buffer for next row
            if (c < DIM) {
                line_buf[0][c] = lbuf1_val;
                line_buf[1][c] = current_pixel;
            }

            // 6. Output Packing (Aligned with PyTorch CHW)
            if (r >= 2 && c >= 2) {
                ap_uint<CH * 9> flat_patch;
                for(int ch = 0; ch < CH; ch++) {
                    #pragma HLS UNROLL
                    for(int wi = 0; wi < 3; wi++) {
                        for(int wj = 0; wj < 3; wj++) {
                            // PyTorch logic: Channel flattens first
                            int offset = (ch * 9) + (wi * 3 + wj);
                            flat_patch[offset] = window[wi][wj][ch];
                        }
                    }
                }
                patch_out.write(flat_patch);
            }
        }
    }

    // Reset counter for next frame (14+2)^2 = 256
    global_pixel_cnt += (DIM + 2) * (DIM + 2);
    if (global_pixel_cnt >= 256) {
        global_pixel_cnt = 0;
    }
}
#endif