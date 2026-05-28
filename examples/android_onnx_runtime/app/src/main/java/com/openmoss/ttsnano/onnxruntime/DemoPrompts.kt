package com.openmoss.ttsnano.onnxruntime

object DemoPrompts {
    // "Welcome to the Android ONNX Runtime demo."
    val ENGLISH_TOKEN_IDS = intArrayOf(
        5322, 300, 280, 351, 391, 373, 543, 10505,
        10505, 11543, 562, 4723, 578, 2140, 10360, 10380,
    )

    // "你好，这是安卓端 ONNX Runtime 示例。"
    val CHINESE_TOKEN_IDS = intArrayOf(
        3985, 10445, 10364, 818, 10651, 12285, 11465, 543,
        10505, 10505, 11543, 562, 4723, 578, 10356, 11088,
        11112, 10382,
    )
}
