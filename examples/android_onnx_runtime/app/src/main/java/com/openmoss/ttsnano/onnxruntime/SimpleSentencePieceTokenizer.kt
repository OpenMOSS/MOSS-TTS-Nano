package com.openmoss.ttsnano.onnxruntime

import java.io.File
import java.text.Normalizer

/**
 * Small SentencePiece tokenizer for the exported Nano tokenizer.model.
 *
 * This keeps the Android example self-contained and avoids native tokenizer
 * dependencies. It intentionally implements only the model fields needed for
 * inference-time encoding.
 */
class SimpleSentencePieceTokenizer private constructor(
    private val pieces: List<Piece>,
    private val normalizerSpec: NormalizerSpec,
) {
    private val unknownId = pieces.indexOfFirst { it.type == PieceType.UNKNOWN }.takeIf { it >= 0 } ?: 0
    private val pieceByText = pieces
        .withIndex()
        .filter { (_, piece) -> piece.type.isTokenizable && piece.text.isNotEmpty() }
        .associateBy { (_, piece) -> piece.text }
    private val piecesByFirstChar = pieces
        .withIndex()
        .filter { (_, piece) -> piece.type.isTokenizable && piece.text.isNotEmpty() }
        .groupBy { (_, piece) -> piece.text[0] }

    fun encode(text: String): IntArray {
        if (text.isBlank()) {
            return IntArray(0)
        }
        val normalized = normalize(text)
        if (normalized.isEmpty()) {
            return IntArray(0)
        }
        if (normalizerSpec.modelType == ModelType.BPE) {
            return encodeBpe(normalized)
        }

        val bestScores = DoubleArray(normalized.length + 1) { Double.NEGATIVE_INFINITY }
        val bestNextIndex = IntArray(normalized.length) { -1 }
        val bestPieceId = IntArray(normalized.length) { unknownId }
        bestScores[normalized.length] = 0.0

        for (index in normalized.length - 1 downTo 0) {
            val candidates = piecesByFirstChar[normalized[index]].orEmpty()
            for ((pieceId, piece) in candidates) {
                if (!normalized.startsWith(piece.text, index)) {
                    continue
                }
                val nextIndex = index + piece.text.length
                val score = piece.score + bestScores[nextIndex]
                if (score > bestScores[index]) {
                    bestScores[index] = score
                    bestNextIndex[index] = nextIndex
                    bestPieceId[index] = pieceId
                }
            }

            if (bestNextIndex[index] < 0) {
                val nextIndex = normalized.offsetByCodePoints(index, 1)
                bestScores[index] = bestScores[nextIndex]
                bestNextIndex[index] = nextIndex
                bestPieceId[index] = unknownId
            }
        }

        val ids = ArrayList<Int>()
        var index = 0
        while (index < normalized.length) {
            ids += bestPieceId[index]
            index = bestNextIndex[index]
        }
        return ids.toIntArray()
    }

    private fun encodeBpe(normalized: String): IntArray {
        val tokens = ArrayList<String>()
        var index = 0
        while (index < normalized.length) {
            val nextIndex = normalized.offsetByCodePoints(index, 1)
            tokens += normalized.substring(index, nextIndex)
            index = nextIndex
        }

        while (tokens.size > 1) {
            var bestIndex = -1
            var bestScore = Float.POSITIVE_INFINITY
            for (tokenIndex in 0 until tokens.lastIndex) {
                val merged = tokens[tokenIndex] + tokens[tokenIndex + 1]
                val candidate = pieceByText[merged]?.value ?: continue
                if (candidate.score < bestScore) {
                    bestScore = candidate.score
                    bestIndex = tokenIndex
                }
            }
            if (bestIndex < 0) {
                break
            }
            tokens[bestIndex] = tokens[bestIndex] + tokens.removeAt(bestIndex + 1)
        }

        return IntArray(tokens.size) { tokenIndex ->
            pieceByText[tokens[tokenIndex]]?.index ?: unknownId
        }
    }

    private fun normalize(text: String): String {
        var normalized = if (normalizerSpec.name.contains("nfkc", ignoreCase = true)) {
            Normalizer.normalize(text, Normalizer.Form.NFKC)
        } else {
            text
        }
        if (normalizerSpec.removeExtraWhitespaces) {
            normalized = normalized.trim().replace(Regex("\\s+"), " ")
        }
        if (normalizerSpec.addDummyPrefix) {
            normalized = " $normalized"
        }
        if (normalizerSpec.escapeWhitespaces) {
            normalized = normalized.replace(Regex("\\s"), "▁")
        }
        return normalized
    }

    companion object {
        fun fromFile(modelFile: File): SimpleSentencePieceTokenizer {
            require(modelFile.isFile) { "Missing tokenizer model: ${modelFile.absolutePath}" }
            return fromModelBytes(modelFile.readBytes())
        }

        fun fromModelBytes(bytes: ByteArray): SimpleSentencePieceTokenizer {
            return SentencePieceModelParser(bytes).parse()
        }
    }

    private data class Piece(
        val text: String,
        val score: Float,
        val type: PieceType,
    )

    private enum class PieceType(val isTokenizable: Boolean) {
        NORMAL(true),
        UNKNOWN(false),
        CONTROL(false),
        USER_DEFINED(true),
        UNUSED(false),
        BYTE(true),
    }

    private data class NormalizerSpec(
        val modelType: ModelType = ModelType.UNIGRAM,
        val name: String = "identity",
        val addDummyPrefix: Boolean = true,
        val removeExtraWhitespaces: Boolean = true,
        val escapeWhitespaces: Boolean = true,
    )

    private enum class ModelType {
        UNIGRAM,
        BPE,
    }

    private class SentencePieceModelParser(private val bytes: ByteArray) {
        fun parse(): SimpleSentencePieceTokenizer {
            val pieces = ArrayList<Piece>()
            var normalizerSpec = NormalizerSpec()
            var modelType = ModelType.UNIGRAM
            val reader = ProtoReader(bytes)
            while (!reader.isAtEnd()) {
                when (val tag = reader.readTag()) {
                    tag(1, WireType.LENGTH_DELIMITED) -> {
                        pieces += parsePiece(reader.readBytes())
                    }
                    tag(2, WireType.LENGTH_DELIMITED) -> {
                        modelType = parseTrainerSpec(reader.readBytes())
                    }
                    tag(3, WireType.LENGTH_DELIMITED) -> {
                        normalizerSpec = parseNormalizerSpec(reader.readBytes())
                    }
                    else -> reader.skip(tag)
                }
            }
            require(pieces.isNotEmpty()) { "No SentencePiece entries found in tokenizer model" }
            return SimpleSentencePieceTokenizer(pieces, normalizerSpec.copy(modelType = modelType))
        }

        private fun parsePiece(pieceBytes: ByteArray): Piece {
            var text = ""
            var score = 0f
            var type = PieceType.NORMAL
            val reader = ProtoReader(pieceBytes)
            while (!reader.isAtEnd()) {
                when (val tag = reader.readTag()) {
                    tag(1, WireType.LENGTH_DELIMITED) -> text = reader.readString()
                    tag(2, WireType.FIXED32) -> score = reader.readFloat()
                    tag(3, WireType.VARINT) -> type = when (reader.readVarint().toInt()) {
                        2 -> PieceType.UNKNOWN
                        3 -> PieceType.CONTROL
                        4 -> PieceType.USER_DEFINED
                        5 -> PieceType.UNUSED
                        6 -> PieceType.BYTE
                        else -> PieceType.NORMAL
                    }
                    else -> reader.skip(tag)
                }
            }
            return Piece(text = text, score = score, type = type)
        }

        private fun parseNormalizerSpec(specBytes: ByteArray): NormalizerSpec {
            var name = "identity"
            var addDummyPrefix = true
            var removeExtraWhitespaces = true
            var escapeWhitespaces = true
            val reader = ProtoReader(specBytes)
            while (!reader.isAtEnd()) {
                when (val tag = reader.readTag()) {
                    tag(1, WireType.LENGTH_DELIMITED) -> name = reader.readString()
                    tag(3, WireType.VARINT) -> addDummyPrefix = reader.readVarint() != 0L
                    tag(4, WireType.VARINT) -> removeExtraWhitespaces = reader.readVarint() != 0L
                    tag(5, WireType.VARINT) -> escapeWhitespaces = reader.readVarint() != 0L
                    else -> reader.skip(tag)
                }
            }
            return NormalizerSpec(
                name = name,
                addDummyPrefix = addDummyPrefix,
                removeExtraWhitespaces = removeExtraWhitespaces,
                escapeWhitespaces = escapeWhitespaces,
            )
        }

        private fun parseTrainerSpec(specBytes: ByteArray): ModelType {
            val reader = ProtoReader(specBytes)
            while (!reader.isAtEnd()) {
                when (val tag = reader.readTag()) {
                    tag(3, WireType.VARINT) -> {
                        return if (reader.readVarint().toInt() == 2) ModelType.BPE else ModelType.UNIGRAM
                    }
                    else -> reader.skip(tag)
                }
            }
            return ModelType.UNIGRAM
        }
    }

    private class ProtoReader(private val bytes: ByteArray) {
        private var offset = 0

        fun isAtEnd(): Boolean = offset >= bytes.size

        fun readTag(): Int = readVarint().toInt()

        fun readVarint(): Long {
            var shift = 0
            var result = 0L
            while (shift < 64) {
                val value = readByte().toInt() and 0xff
                result = result or ((value and 0x7f).toLong() shl shift)
                if (value and 0x80 == 0) {
                    return result
                }
                shift += 7
            }
            error("Invalid varint in tokenizer model")
        }

        fun readBytes(): ByteArray {
            val length = readVarint().toInt()
            require(length >= 0 && offset + length <= bytes.size) { "Invalid length-delimited field" }
            return bytes.copyOfRange(offset, offset + length).also {
                offset += length
            }
        }

        fun readString(): String = readBytes().toString(Charsets.UTF_8)

        fun readFloat(): Float {
            require(offset + 4 <= bytes.size) { "Invalid fixed32 field" }
            val bits = (bytes[offset].toInt() and 0xff) or
                ((bytes[offset + 1].toInt() and 0xff) shl 8) or
                ((bytes[offset + 2].toInt() and 0xff) shl 16) or
                ((bytes[offset + 3].toInt() and 0xff) shl 24)
            offset += 4
            return java.lang.Float.intBitsToFloat(bits)
        }

        fun skip(tag: Int) {
            when (tag and 0x7) {
                WireType.VARINT.id -> readVarint()
                WireType.FIXED64.id -> skipBytes(8)
                WireType.LENGTH_DELIMITED.id -> skipBytes(readVarint().toInt())
                WireType.FIXED32.id -> skipBytes(4)
                else -> error("Unsupported protobuf wire type: ${tag and 0x7}")
            }
        }

        private fun readByte(): Byte {
            require(offset < bytes.size) { "Unexpected end of tokenizer model" }
            return bytes[offset++]
        }

        private fun skipBytes(count: Int) {
            require(count >= 0 && offset + count <= bytes.size) { "Invalid protobuf field length" }
            offset += count
        }
    }
}

private enum class WireType(val id: Int) {
    VARINT(0),
    FIXED64(1),
    LENGTH_DELIMITED(2),
    FIXED32(5),
}

private fun tag(fieldNumber: Int, wireType: WireType): Int {
    return (fieldNumber shl 3) or wireType.id
}
