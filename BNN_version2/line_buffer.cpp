#include <ap_int.h>
#include <hls_stream.h>
#include <ap_int.h>
#include <hls_stream.h>

/**
 * Binary valid 3x3 line buffer
 * Input:  28×28 = 784 pixels
 * Output: 26×26 = 676 windows
 */
void line_buffer_bin_3x3(
    hls::stream<ap_uint<1>> &pixel_in,
    hls::stream<ap_uint<9>> &window_out
) {
    static ap_uint<1> line_buf[2][28];
#pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1

    static ap_uint<1> window[3][3];
#pragma HLS ARRAY_PARTITION variable=window complete

    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
#pragma HLS PIPELINE II=1
            ap_uint<1> new_pixel = pixel_in.read();

            // Shift window left
            for (int i = 0; i < 3; i++) {
                window[i][0] = window[i][1];
                window[i][1] = window[i][2];
            }

            // Insert new column
            window[0][2] = (r >= 2) ? line_buf[0][c] : ap_uint<1>(0);
            window[1][2] = (r >= 1) ? line_buf[1][c] : ap_uint<1>(0);
            window[2][2] = new_pixel;

            // Update line buffers
            line_buf[0][c] = line_buf[1][c];
            line_buf[1][c] = new_pixel;

            // Output only VALID 3x3 windows
            if (r >= 2 && c >= 2) {
                ap_uint<9> w;
                w[0] = window[0][0];
                w[1] = window[0][1];
                w[2] = window[0][2];
                w[3] = window[1][0];
                w[4] = window[1][1];
                w[5] = window[1][2];
                w[6] = window[2][0];
                w[7] = window[2][1];
                w[8] = window[2][2];
                window_out.write(w);
            }
        }
    }
}
