#ifndef BNN_CONFIG_H
#define BNN_CONFIG_H

// ================= CONV1 =================
#define CONV1_ROWS     32
#define CONV1_COLS     9
#define CONV1_NNZ      132
#define CONV1_POP_W    10
#define CONV1_THRESH_W 10

// ================= CONV2 =================
#define CONV2_CH       CONV1_ROWS   // FIX
#define CONV2_ROWS     64
#define CONV2_COLS     288
#define CONV2_NNZ      9719
#define CONV2_POP_W    10
#define CONV2_THRESH_W 10

// ================= CONV3 =================
#define CONV3_CH       CONV2_ROWS   // FIX
#define CONV3_ROWS     128
#define CONV3_COLS     576
#define CONV3_NNZ      34212
#define CONV3_POP_W    11
#define CONV3_THRESH_W 11

// ================= FC1 =================
#define FC1_COLS       CONV3_ROWS   // 128
#define FC1_ROWS       256
#define FC1_NNZ        16456
#define FC1_POP_W      10
#define FC1_THRESH_W   10

// ================= FC2 =================
#define FC2_COLS       FC1_ROWS     // 256
#define FC2_ROWS       9
#define FC2_NNZ        1274
#endif
