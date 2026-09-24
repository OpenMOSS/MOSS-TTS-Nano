package com.openmoss.ttsnano.onnxruntime

import org.junit.Assert.assertArrayEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

class SimpleSentencePieceTokenizerTest {
    @Test
    fun encodesTextWithWhitespacePrefixAndNfkcNormalization() {
        val tokenizer = SimpleSentencePieceTokenizer.fromModelBytes(
            buildModel(
                piece("<unk>", 0f, 2),
                piece("▁Hello", -0.1f),
                piece("▁world", -0.1f),
                piece("!", -0.1f),
                piece("▁你", -0.1f),
                piece("好", -0.1f),
                piece(",", -0.1f),
                piece("世界", -0.1f),
            ),
        )

        assertArrayEquals(intArrayOf(1, 2, 3), tokenizer.encode("Hello world!"))
        assertArrayEquals(intArrayOf(4, 5, 6, 7, 3), tokenizer.encode("你好，世界！"))
    }

    @Test
    fun blankTextEncodesToNoTokens() {
        val tokenizer = SimpleSentencePieceTokenizer.fromModelBytes(
            buildModel(
                piece("<unk>", 0f, 2),
                piece("▁", -0.1f),
            ),
        )

        assertArrayEquals(IntArray(0), tokenizer.encode(""))
        assertArrayEquals(IntArray(0), tokenizer.encode("  \n\t  "))
    }

    @Test
    fun prefersTheHighestScoredUnigramPath() {
        val tokenizer = SimpleSentencePieceTokenizer.fromModelBytes(
            buildModel(
                piece("<unk>", 0f, 2),
                piece("▁a", -2.0f),
                piece("▁ab", -0.1f),
                piece("b", -0.1f),
            ),
        )

        assertArrayEquals(intArrayOf(2), tokenizer.encode("ab"))
    }

    @Test
    fun bpeModelMergesPairsByRank() {
        val tokenizer = SimpleSentencePieceTokenizer.fromModelBytes(
            buildModel(
                2,
                piece("<unk>", 0f, 2),
                piece("▁", 0f),
                piece("H", 0f),
                piece("e", 0f),
                piece("l", 0f),
                piece("o", 0f),
                piece("▁H", -1f),
                piece("▁He", -2f),
                piece("ll", -3f),
                piece("▁Hell", -4f),
                piece("▁Hello", -5f),
            ),
        )

        assertArrayEquals(intArrayOf(10), tokenizer.encode("Hello"))
    }

    @Test
    fun encodesRealMossTokenizerWhenModelPathIsProvided() {
        val modelPath = System.getenv("MOSS_TOKENIZER_MODEL").orEmpty()
        assumeTrue("Set MOSS_TOKENIZER_MODEL to run this optional integration test", modelPath.isNotBlank())
        val tokenizer = SimpleSentencePieceTokenizer.fromFile(File(modelPath))

        assertArrayEquals(intArrayOf(7026, 1177, 11449), tokenizer.encode("Hello world!"))
        assertArrayEquals(intArrayOf(3985, 10445, 10364, 1260, 11449), tokenizer.encode("你好，世界！"))
    }

    private fun buildModel(vararg pieces: ByteArray): ByteArray {
        return buildModel(1, *pieces)
    }

    private fun buildModel(modelType: Int, vararg pieces: ByteArray): ByteArray {
        return proto {
            pieces.forEach { bytes ->
                fieldBytes(1, bytes)
            }
            fieldBytes(2) {
                fieldVarint(3, modelType)
            }
            fieldBytes(3) {
                fieldString(1, "nmt_nfkc")
                fieldVarint(3, 1)
                fieldVarint(4, 1)
                fieldVarint(5, 1)
            }
        }
    }

    private fun piece(text: String, score: Float, type: Int = 1): ByteArray {
        return proto {
            fieldString(1, text)
            fieldFloat(2, score)
            fieldVarint(3, type)
        }
    }

    private fun proto(block: ProtoWriter.() -> Unit): ByteArray {
        return ProtoWriter().apply(block).toByteArray()
    }

}

private class ProtoWriter {
    private val bytes = ArrayList<Byte>()

    fun fieldString(fieldNumber: Int, value: String) {
        fieldBytes(fieldNumber, value.toByteArray(Charsets.UTF_8))
    }

    fun fieldBytes(fieldNumber: Int, value: ByteArray) {
        writeVarint((fieldNumber shl 3) or 2)
        writeVarint(value.size)
        value.forEach { bytes += it }
    }

    fun fieldBytes(fieldNumber: Int, block: ProtoWriter.() -> Unit) {
        fieldBytes(fieldNumber, ProtoWriter().apply(block).toByteArray())
    }

    fun fieldVarint(fieldNumber: Int, value: Int) {
        writeVarint((fieldNumber shl 3) or 0)
        writeVarint(value)
    }

    fun fieldFloat(fieldNumber: Int, value: Float) {
        writeVarint((fieldNumber shl 3) or 5)
        val bits = java.lang.Float.floatToIntBits(value)
        bytes += (bits and 0xff).toByte()
        bytes += ((bits ushr 8) and 0xff).toByte()
        bytes += ((bits ushr 16) and 0xff).toByte()
        bytes += ((bits ushr 24) and 0xff).toByte()
    }

    fun toByteArray(): ByteArray = bytes.toByteArray()

    private fun writeVarint(value: Int) {
        var remaining = value
        while (remaining and 0x7f.inv() != 0) {
            bytes += ((remaining and 0x7f) or 0x80).toByte()
            remaining = remaining ushr 7
        }
        bytes += remaining.toByte()
    }
}
