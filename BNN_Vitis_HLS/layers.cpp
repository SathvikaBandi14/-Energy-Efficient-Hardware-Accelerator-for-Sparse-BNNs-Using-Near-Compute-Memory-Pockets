/***#include <ap_int.h>
#include <hls_stream.h>
#include "binarize.h"
#include "line_buffer.h"
#include "line_buffer_multi.h"
#include "pe_unit.h"
#include "bnn_config.h"

#include "conv1_data.h"
#include "conv2_data.h"
#include "conv3_data.h"
#include "fc1_data.h"
#include "fc2_data.h"

// CONV1 (Stays 2 arguments because bias is hard-coded 0.0f inside)


// ============================================================
// CONV1 - 28x28
// ============================================================
void compute_conv1(hls::stream<ap_uint<8>> &in_pixels, hls::stream<ap_uint<CONV1_ROWS>> &out_bits) {
    hls::stream<ap_uint<1>> bin_stream("bin1");
    hls::stream<ap_uint<CONV1_COLS>> patch_stream("patch1");
    hls::stream<ap_uint<CONV1_ROWS>> raw_conv_out("raw_conv_out");

    // DEPTH IS CRITICAL: Must be able to buffer enough for the loop to start
    #pragma HLS STREAM variable=bin_stream depth=32
    #pragma HLS STREAM variable=patch_stream depth=32
    #pragma HLS STREAM variable=raw_conv_out depth=784 

    binarize(in_pixels, bin_stream);
    line_buffer_bin_3x3(bin_stream, patch_stream);
    
    sparse_bnn_layer<CONV1_ROWS, CONV1_COLS, CONV1_NNZ, 784>(
        patch_stream, raw_conv_out, conv1_weights, conv1_cols,
        conv1_row_ptr, conv1_thresholds, 0.0f
    );

    for(int p=0; p < 784; p++) {
        #pragma HLS PIPELINE II=1
        ap_uint<CONV1_ROWS> val = raw_conv_out.read();
        out_bits.write(val);

        #ifndef __SYNTHESIS__
            if (p % 28 == 0) printf("\n");
            // Check Channel 0 to see if the '6' shape exists
            printf("%c", val[0] ? '#' : '.'); 
        #endif
    }
}

// ============================================================
// CONV2 - 14x14
// ============================================================
// ============================================================
// CONV2 - 14x14
// ============================================================
void compute_conv2(
    hls::stream<ap_uint<CONV2_CH>> &in_pixels, 
    hls::stream<ap_uint<CONV2_ROWS>> &out_bits, 
    float bias_shift
) {
    // Internal streams
    hls::stream<ap_uint<CONV2_COLS>> patch_stream("patch2");
    hls::stream<ap_uint<CONV2_ROWS>> raw_out("raw_out");
    hls::stream<ap_uint<CONV2_CH>> in_pixels_internal("in_pixels_internal");

    #pragma HLS STREAM variable=patch_stream depth=32
    #pragma HLS STREAM variable=raw_out depth=196
    #pragma HLS STREAM variable=in_pixels_internal depth=196

    // ---------------------------------------------------------
    // 1. SNOOP INPUT (Bits coming from Pool1/Conv1)
    // ---------------------------------------------------------
    #ifndef __SYNTHESIS__
    printf("\n--- CONV2 INPUT BITS (14x14 grid, Channels 0-7) ---");
    #endif

    for(int i = 0; i < 196; i++) {
        ap_uint<CONV2_CH> val = in_pixels.read();
        
        #ifndef __SYNTHESIS__
        if (i % 14 == 0) printf("\n");
        // Print '#' if any of the first 8 channels are active
        printf("%c", (val & 0xFF) ? '#' : '.');
        #endif

        // Pass data to the internal stream for the line buffer
        in_pixels_internal.write(val);
    }

    // ---------------------------------------------------------
    // 2. LINE BUFFER & SPARSE ENGINE
    // ---------------------------------------------------------
    // This generates 3x3 patches (9 pixels * 32 channels = 288 bits)
    line_buffer_multi_padded<CONV2_CH, 14>(in_pixels_internal, patch_stream);

    // This performs the binary convolution
    sparse_bnn_layer<CONV2_ROWS, CONV2_COLS, CONV2_NNZ, 196>(
        patch_stream, raw_out, conv2_weights, conv2_cols,
        conv2_row_ptr, conv2_thresholds, bias_shift
    );

    // ---------------------------------------------------------
    // 3. SNOOP OUTPUT (Bits going to Pool2)
    // ---------------------------------------------------------
    #ifndef __SYNTHESIS__
    printf("\n\n--- CONV2 OUTPUT BITS (14x14 grid, Channels 0-7) ---");
    #endif

    for(int i = 0; i < 196; i++) {
        // We MUST read from the raw_out stream produced by the sparse engine
        ap_uint<CONV2_ROWS> val = raw_out.read();
        
        // Forward the data to the next layer in the accelerator
        out_bits.write(val);

        #ifndef __SYNTHESIS__
        if (i % 14 == 0) printf("\n");
        // Aggregate check on the first 8 output channels
        printf("%c", (val & 0xFF) ? '#' : '.'); 
        #endif
    }
    
    #ifndef __SYNTHESIS__
    printf("\n----------------------------------------------------\n");
    #endif
}
// ============================================================
// CONV3 - 7x7
// ============================================================
void compute_conv3(hls::stream<ap_uint<CONV3_CH>> &in_pixels, hls::stream<ap_uint<CONV3_ROWS>> &out_bits, float bias_shift) {
    hls::stream<ap_uint<CONV3_COLS>> patch_stream("patch3");
    hls::stream<ap_uint<CONV3_ROWS>> raw_out("raw_out");

    #pragma HLS STREAM variable=patch_stream depth=32
    #pragma HLS STREAM variable=raw_out depth=49

    line_buffer_multi_padded<CONV3_CH, 7>(in_pixels, patch_stream);
    
    sparse_bnn_layer<CONV3_ROWS, CONV3_COLS, CONV3_NNZ, 49>(
        patch_stream, raw_out, conv3_weights, conv3_cols,
        conv3_row_ptr, conv3_thresholds, bias_shift
    );

    #ifndef __SYNTHESIS__
    printf("\n--- CONV3 MAP (Aggregate Activity) ---\n");
    #endif

    for(int i = 0; i < 49; i++) {
        ap_uint<128> val = raw_out.read();
        out_bits.write(val);

        #ifndef __SYNTHESIS__
        if (i % 7 == 0) printf("\n");
        // Print '#' if ANY channel is active at this pixel
        bool any_bit = false;
        for(int b=0; b<128; b++) if(val[b]) any_bit = true;
        printf("%c", any_bit ? '#' : '.'); 
        #endif
    }
}
// ============================================================
// FC1 & FC2
// ============================================================
void compute_fc1(hls::stream<ap_uint<FC1_COLS>> &in_vector, hls::stream<ap_uint<FC1_ROWS>> &out_bits, float bias_shift) {
    sparse_bnn_layer<FC1_ROWS, FC1_COLS, FC1_NNZ, 1>(
        in_vector, out_bits, fc1_weights, fc1_cols, fc1_row_ptr, fc1_thresholds, bias_shift
    );
}

void compute_fc2(hls::stream<ap_uint<FC2_COLS>> &in_vector, hls::stream<ap_uint<FC2_ROWS>> &out_bits, float bias_shift) {
    sparse_bnn_layer<FC2_ROWS, FC2_COLS, FC2_NNZ, 1>(
        in_vector, out_bits, fc2_weights, fc2_cols, fc2_row_ptr, fc2_thresholds, bias_shift
    );
}
***/
#include <ap_int.h>
#include <hls_stream.h>

#include "binarize.h"
#include "line_buffer.h"
#include "line_buffer_multi.h"
#include "pe_unit.h"
#include "bnn_config.h"
#include "pe_unit.h"

// ================= DATA =================
#include "conv1_data.h"
#include "conv1_thresholds_int.h"

#include "conv2_data.h"
#include "conv2_thresholds_int.h"

#include "conv3_data.h"
#include "conv3_thresholds_int.h"

#include "fc1_data.h"
#include "fc1_thresholds_int.h"

#include "fc2_data.h"
#include "fc2_thresholds_int.h"

// ============================================================
// CONV1 - 28x28
// ============================================================
void compute_conv1(
    hls::stream<ap_uint<8>> &in_pixels,
    hls::stream<ap_uint<CONV1_ROWS>> &out_bits
) {
    hls::stream<ap_uint<1>> bin_stream("bin1");
    hls::stream<ap_uint<CONV1_COLS>> patch_stream("patch1");
    hls::stream<ap_uint<CONV1_ROWS>> raw_conv_out("raw_conv_out");

#pragma HLS STREAM variable=bin_stream depth=32
#pragma HLS STREAM variable=patch_stream depth=32
#pragma HLS STREAM variable=raw_conv_out depth=784

    binarize(in_pixels, bin_stream);
    line_buffer_bin_3x3(bin_stream, patch_stream);

    sparse_bnn_layer<
    CONV1_ROWS,
    CONV1_COLS,
    CONV1_NNZ,
    784,
    CONV1_POP_W,
    CONV1_THRESH_W,
    3   // THRESH_IN_W (ap_uint<3>)
>(
    patch_stream,
    raw_conv_out,
    conv1_weights,
    conv1_cols,
    conv1_row_ptr,
    conv1_thresholds_int
);

    for (int p = 0; p < 784; p++) {
#pragma HLS PIPELINE II=1
        ap_uint<CONV1_ROWS> val = raw_conv_out.read();
        out_bits.write(val);

#ifndef __SYNTHESIS__
        if (p % 28 == 0) printf("\n");
        printf("%c", val[0] ? '#' : '.');
#endif
    }
}

// ============================================================
// CONV2 - 14x14
// ============================================================
void compute_conv2(
    hls::stream<ap_uint<CONV2_CH>> &in_pixels,
    hls::stream<ap_uint<CONV2_ROWS>> &out_bits
) {
    hls::stream<ap_uint<CONV2_COLS>> patch_stream("patch2");
    hls::stream<ap_uint<CONV2_ROWS>> raw_out("raw_out");
    hls::stream<ap_uint<CONV2_CH>> in_pixels_internal("in_pixels_internal");

#pragma HLS STREAM variable=patch_stream depth=32
#pragma HLS STREAM variable=raw_out depth=196
#pragma HLS STREAM variable=in_pixels_internal depth=196

#ifndef __SYNTHESIS__
    printf("\n--- CONV2 INPUT BITS (14x14) ---\n");
#endif

    for (int i = 0; i < 196; i++) {
        ap_uint<CONV2_CH> val = in_pixels.read();
        in_pixels_internal.write(val);

#ifndef __SYNTHESIS__
        if (i % 14 == 0) printf("\n");
        printf("%c", (val & 0xFF) ? '#' : '.');
#endif
    }

    line_buffer_multi_padded<CONV2_CH, 14>(
        in_pixels_internal,
        patch_stream
    );

    sparse_bnn_layer<
    CONV2_ROWS,
    CONV2_COLS,
    CONV2_NNZ,
    196,
    CONV2_POP_W,     // 9  (math domain)
    CONV2_THRESH_W,  // 9  (math domain)
    8                // 👈 THRESH_IN_W (actual header width!)
>(
    patch_stream,
    raw_out,
    conv2_weights,
    conv2_cols,
    conv2_row_ptr,
    conv2_thresholds_int
);


#ifndef __SYNTHESIS__
    printf("\n\n--- CONV2 OUTPUT BITS (14x14) ---\n");
#endif

    for (int i = 0; i < 196; i++) {
        ap_uint<CONV2_ROWS> val = raw_out.read();
        out_bits.write(val);

#ifndef __SYNTHESIS__
        if (i % 14 == 0) printf("\n");
        printf("%c", (val & 0xFF) ? '#' : '.');
#endif
    }
}

// ============================================================
// CONV3 - 7x7
// ============================================================
void compute_conv3(
    hls::stream<ap_uint<CONV3_CH>> &in_pixels,
    hls::stream<ap_uint<CONV3_ROWS>> &out_bits
) {
    hls::stream<ap_uint<CONV3_COLS>> patch_stream("patch3");
    hls::stream<ap_uint<CONV3_ROWS>> raw_out("raw_out");

#pragma HLS STREAM variable=patch_stream depth=32
#pragma HLS STREAM variable=raw_out depth=49

    line_buffer_multi_padded<CONV3_CH, 7>(
        in_pixels,
        patch_stream
    );

    sparse_bnn_layer<
    CONV3_ROWS,
    CONV3_COLS,
    CONV3_NNZ,
    49,
    CONV3_POP_W,
    CONV3_THRESH_W,
    9
>(
    patch_stream,
    raw_out,
    conv3_weights,
    conv3_cols,
    conv3_row_ptr,
    conv3_thresholds_int
);

#ifndef __SYNTHESIS__
    printf("\n--- CONV3 OUTPUT (7x7) ---\n");
#endif

    for (int i = 0; i < 49; i++) {
        ap_uint<CONV3_ROWS> val = raw_out.read();
        out_bits.write(val);

#ifndef __SYNTHESIS__
        if (i % 7 == 0) printf("\n");
        bool any = false;
        for (int b = 0; b < CONV3_ROWS; b++)
            if (val[b]) any = true;
        printf("%c", any ? '#' : '.');
#endif
    }
}

// ============================================================
// FC1
// ============================================================
void compute_fc1(
    hls::stream<ap_uint<FC1_COLS>> &in_vector,
    hls::stream<ap_uint<FC1_ROWS>> &out_bits
) {
    sparse_bnn_layer<
        FC1_ROWS,
        FC1_COLS,
        FC1_NNZ,
        1,
        FC1_POP_W,
        FC1_THRESH_W,
        10
    >(
        in_vector,
        out_bits,
        fc1_weights,
        fc1_cols,
        fc1_row_ptr,
        fc1_thresholds_int
    );
}

// ============================================================
// FC2
// ============================================================
void compute_fc2(
    hls::stream<ap_uint<FC2_COLS>> &in_vector,
    hls::stream<ap_uint<FC2_ROWS>> &out_bits
) {
    sparse_bnn_layer<
        FC2_ROWS,
        FC2_COLS,
        FC2_NNZ,
        1,
        FC2_POP_W,
        FC2_THRESH_W,
        8
    >(
        in_vector,
        out_bits,
        fc2_weights,
        fc2_cols,
        fc2_row_ptr,
        fc2_thresholds_int
    );
}
