#ifndef BNN_CONFIG_H
#define BNN_CONFIG_H

// ==========================
// Convolution Layer 1
// ==========================
#define CONV1_CH        1
#define CONV1_ROWS      32
#define CONV1_COLS      9
#define CONV1_NNZ       261

#define CONV1_POP_W     4
#define CONV1_THRESH_W  4

// ==========================
// Convolution Layer 2
// ==========================
#define CONV2_CH        32
#define CONV2_ROWS      64
#define CONV2_COLS      288
#define CONV2_NNZ       16704

#define CONV2_POP_W     9
#define CONV2_THRESH_W  9

// ==========================
// Convolution Layer 3
// ==========================
#define CONV3_CH        64
#define CONV3_ROWS      128
#define CONV3_COLS      576
#define CONV3_NNZ       66816

#define CONV3_POP_W     10
#define CONV3_THRESH_W  10

// ==========================
// Fully Connected Layer 1
// ==========================
#define FC1_ROWS        256
#define FC1_COLS        1152
#define FC1_NNZ         266112

#define FC1_POP_W       11
#define FC1_THRESH_W    11

// ==========================
// Fully Connected Layer 2
// ==========================
#define FC2_ROWS        10
#define FC2_COLS        256
#define FC2_NNZ         2304

#define FC2_POP_W       9
#define FC2_THRESH_W    9

#endif
