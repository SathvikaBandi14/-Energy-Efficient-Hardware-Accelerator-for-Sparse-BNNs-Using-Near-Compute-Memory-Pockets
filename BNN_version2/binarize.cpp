#include <ap_int.h>
#include <hls_stream.h>

void binarize(
    hls::stream<ap_uint<8>> &in_stream,
    hls::stream<ap_uint<1>> &out_stream
) {
    #ifndef __SYNTHESIS__
    printf("--- Hardware Binarizer View ---\n");
    #endif

    for (int i = 0; i < 784; i++) {
        #pragma HLS PIPELINE II=1
        ap_uint<8> pixel = in_stream.read();
        ap_uint<1> bit = (pixel >= 127) ? (ap_uint<1>)1 : (ap_uint<1>)0;

        #ifndef __SYNTHESIS__
        // Print # for white pixels, space for black
        printf("%s", bit ? "#" : ".");
        // Every 28 pixels, print a newline to show the row
        if ((i + 1) % 28 == 0) printf("\n");
        #endif

        out_stream.write(bit);
    }
}