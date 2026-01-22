/**
 * Flattening Module: 128 channels * 1x1 spatial = 128 bits
 * Since spatial is 1x1, no transposition or buffering is needed.
 */
#include <ap_int.h>
#include <hls_stream.h>
void flatten_to_128(
    hls::stream<ap_uint<128>> &in_pool3, 
    hls::stream<ap_uint<128>> &out_flatten
) {
    // This will now block and wait for the data from Global Maxpool
    ap_uint<128> single_pixel = in_pool3.read();
    
    // Pass the 128-bit feature vector to FC1
    out_flatten.write(single_pixel);
}