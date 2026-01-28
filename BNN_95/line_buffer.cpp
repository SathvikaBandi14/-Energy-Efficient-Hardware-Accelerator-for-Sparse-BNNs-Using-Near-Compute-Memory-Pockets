#include <ap_int.h>
#include <hls_stream.h>
#include <ap_int.h>
#include <hls_stream.h>
void line_buffer_bin_3x3(
    hls::stream<ap_uint<1>> &pixel_in,
    hls::stream<ap_uint<9>> &window_out
) {
    static ap_uint<1> line_buf[2][28];
    static ap_uint<1> window[3][3];

    // To maintain 28x28, we iterate from -1 to 28 (total 30) or handle boundaries
    for (int r = 0; r < 30; r++) {
        for (int c = 0; c < 30; c++) {
            #pragma HLS PIPELINE II=1
            
            ap_uint<1> new_pixel = 0; // Default zero padding
            // Only read from stream if within the actual 28x28 image boundaries
            if (r >= 1 && r < 29 && c >= 1 && c < 29) {
                new_pixel = pixel_in.read();
            }

            // ... (Shift window logic stays the same) ...

            // Output windows for every pixel to keep 28x28 output
            if (r >= 2 && c >= 2) {
                ap_uint<9> w;
                // Pack window[3][3] into w...
                window_out.write(w);
            }
        }
    }
}