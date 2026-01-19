#include <ap_int.h>
#include <hls_stream.h>

#define IMG_WIDTH 28 // Example for MNIST
#define KERNEL_SIZE 3

void line_buffer_bin_3x3(
    hls::stream<ap_uint<1>> &pixel_in,
    hls::stream<ap_uint<9>> &window_out
) {
    static ap_uint<1> line_buf[2][28];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1

    static ap_uint<1> window[3][3];
    #pragma HLS ARRAY_PARTITION variable=window complete

    // Clear window for every NEW image call to be safe
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++) window[i][j] = 0;

    // We run 30x30 to handle the padding edges (padding=1)
    for (int r = 0; r < 30; r++) {
        for (int c = 0; c < 30; c++) {
            #pragma HLS PIPELINE II=1
            
            ap_uint<1> new_pixel = 0;
            // Only read if we are within the 28x28 bounds
            // Note: Offset by 1 for padding=1
            if (r >= 1 && r < 29 && c >= 1 && c < 29) {
                new_pixel = pixel_in.read();
            }

            // 1. Shift Window left
            for (int i = 0; i < 3; i++) {
                window[i][0] = window[i][1];
                window[i][1] = window[i][2];
            }

            // 2. Fill right column
            // Top of window comes from line_buf[0]
            // Middle comes from line_buf[1]
            // Bottom is the new pixel entering
            window[0][2] = (c > 0 && c <= 28) ? line_buf[0][c-1] : (ap_uint<1>)0;
            window[1][2] = (c > 0 && c <= 28) ? line_buf[1][c-1] : (ap_uint<1>)0;
            window[2][2] = new_pixel;

            // 3. Shift Line Buffer
            if (c > 0 && c <= 28) {
                line_buf[0][c-1] = line_buf[1][c-1];
                line_buf[1][c-1] = new_pixel;
            }

            // 4. Output only the 28x28 valid window positions
            // This happens once the window is "centered"
            if (r >= 2 && c >= 2) {
                ap_uint<9> flattened_window;
                flattened_window[0] = window[0][0];
                flattened_window[1] = window[0][1];
                flattened_window[2] = window[0][2];
                flattened_window[3] = window[1][0];
                flattened_window[4] = window[1][1];
                flattened_window[5] = window[1][2];
                flattened_window[6] = window[2][0];
                flattened_window[7] = window[2][1];
                flattened_window[8] = window[2][2];
                window_out.write(flattened_window);
            }
        }
    }
}