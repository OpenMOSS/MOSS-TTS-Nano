package com.openmoss.ttsnano.onnxruntime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxTensorLike
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import kotlin.math.min
import org.json.JSONArray
import org.json.JSONObject

class MossOnnxDemoEngine(
    private val modelRoot: File,
    private val outputDir: File,
    private val cpuThreads: Int = 2,
) : Closeable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val manifestPath = resolveManifestPath(modelRoot)
    private val manifestDir = manifestPath.parentFile ?: modelRoot
    private val manifest = ModelManifest.fromJson(readJson(manifestPath))
    private val ttsMetaPath = resolveManifestRelativePath(manifest.modelFiles.ttsMeta)
    private val codecMetaPath = resolveManifestRelativePath(manifest.modelFiles.codecMeta)
    private val ttsMeta = TtsMeta.fromJson(readJson(ttsMetaPath))
    private val codecMeta = CodecMeta.fromJson(readJson(codecMetaPath))
    private val ttsDir = ttsMetaPath.parentFile ?: manifestDir
    private val codecDir = codecMetaPath.parentFile ?: manifestDir
    private val sessionOptions = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setIntraOpNumThreads(cpuThreads.coerceAtLeast(1))
        setInterOpNumThreads(1)
    }
    private val prefillSession = createSession(File(ttsDir, ttsMeta.files.prefill))
    private val decodeSession = createSession(File(ttsDir, ttsMeta.files.decodeStep))
    private val localFixedFrameSession = createSession(File(ttsDir, ttsMeta.files.localFixedSampledFrame))
    private val codecDecodeSession = createSession(File(codecDir, codecMeta.files.decodeFull))

    fun synthesize(
        textTokenIds: IntArray,
        outputFile: File = File(outputDir, "moss_tts_nano_android.wav"),
        voice: String = "Junhao",
        maxFrames: Int = 160,
        seed: Long = 1234L,
    ): SynthesisResult {
        require(textTokenIds.isNotEmpty()) { "textTokenIds must not be empty" }
        val startedAt = System.currentTimeMillis()
        val inputRows = buildInputRows(textTokenIds, voice)
        val prefillResult = runPrefill(inputRows)
        val audioTokens = runDecode(prefillResult, maxFrames, seed)
        val pcm = decodeAudioTokens(audioTokens)
        val sampleRate = codecMeta.codecConfig.sampleRate
        writeWavMono(pcm, sampleRate, outputFile)
        val elapsedMs = System.currentTimeMillis() - startedAt
        return SynthesisResult(
            outputFile = outputFile,
            generatedFrames = audioTokens.size,
            sampleRate = sampleRate,
            durationMs = (pcm.size.toDouble() / sampleRate * 1000.0).toLong(),
            elapsedMs = elapsedMs,
        )
    }

    override fun close() {
        codecDecodeSession.close()
        localFixedFrameSession.close()
        decodeSession.close()
        prefillSession.close()
        sessionOptions.close()
    }

    private fun createSession(modelFile: File): OrtSession {
        require(modelFile.isFile) { "Missing ONNX file: ${modelFile.absolutePath}" }
        return env.createSession(modelFile.absolutePath, sessionOptions)
    }

    private fun resolveManifestRelativePath(relativePath: String): File {
        val direct = File(manifestDir, relativePath).canonicalFile
        if (direct.exists()) {
            return direct
        }
        val alias = relativePath
            .replace("MOSS-TTS-Nano-ONNX-CPU", "MOSS-TTS-Nano-100M-ONNX")
            .replace("MOSS-Audio-Tokenizer-Nano-ONNX-CPU", "MOSS-Audio-Tokenizer-Nano-ONNX")
        return File(manifestDir, alias).canonicalFile
    }

    private fun buildInputRows(textTokenIds: IntArray, voice: String): InputRows {
        val cfg = manifest.ttsConfig
        val rowWidth = cfg.nVq + 1
        val promptAudioCodes = selectBuiltinVoicePromptAudioCodes(voice)
        val prefixTokens = manifest.promptTemplates.userPromptPrefixTokenIds + cfg.audioStartTokenId
        val suffixTokens = intArrayOf(cfg.audioEndTokenId) +
            manifest.promptTemplates.userPromptAfterReferenceTokenIds +
            textTokenIds +
            manifest.promptTemplates.assistantPromptPrefixTokenIds +
            intArrayOf(cfg.audioStartTokenId)
        val rows = ArrayList<IntArray>()
        rows += buildTextRows(prefixTokens, cfg, rowWidth)
        rows += buildAudioRows(promptAudioCodes, cfg, rowWidth)
        rows += buildTextRows(suffixTokens, cfg, rowWidth)
        return InputRows(rows.toTypedArray(), IntArray(rows.size) { 1 })
    }

    private fun buildTextRows(tokens: IntArray, cfg: TtsConfig, rowWidth: Int): List<IntArray> {
        return tokens.map { token ->
            IntArray(rowWidth) { index -> if (index == 0) token else cfg.audioPadTokenId }
        }
    }

    private fun buildAudioRows(audioCodes: List<IntArray>, cfg: TtsConfig, rowWidth: Int): List<IntArray> {
        return audioCodes.map { codeRow ->
            IntArray(rowWidth) { index ->
                when {
                    index == 0 -> cfg.audioUserSlotTokenId
                    index - 1 < min(codeRow.size, cfg.nVq) -> codeRow[index - 1]
                    else -> cfg.audioPadTokenId
                }
            }
        }
    }

    private fun selectBuiltinVoicePromptAudioCodes(voice: String): List<IntArray> {
        val selected = manifest.builtinVoices.firstOrNull {
            it.voice == voice && it.promptAudioCodes.isNotEmpty()
        } ?: manifest.builtinVoices.firstOrNull { it.promptAudioCodes.isNotEmpty() }
        return selected?.promptAudioCodes
            ?: error("No builtin voice prompt_audio_codes found in ${manifestPath.absolutePath}")
    }

    private fun runPrefill(inputRows: InputRows): PrefillResult {
        val seqLen = inputRows.inputIds.size
        val rowWidth = inputRows.inputIds[0].size
        val inputIdsFlat = IntArray(seqLen * rowWidth)
        var offset = 0
        for (row in inputRows.inputIds) {
            for (value in row) {
                inputIdsFlat[offset++] = value
            }
        }
        OnnxTensor.createTensor(
            env,
            IntBuffer.wrap(inputIdsFlat),
            longArrayOf(1, seqLen.toLong(), rowWidth.toLong()),
        ).use { inputIdsTensor ->
            OnnxTensor.createTensor(
                env,
                IntBuffer.wrap(inputRows.attentionMask),
                longArrayOf(1, seqLen.toLong()),
            ).use { maskTensor ->
                val outputs = prefillSession.run(
                    mapOf(
                        "input_ids" to inputIdsTensor,
                        "attention_mask" to maskTensor,
                    ),
                )
                return PrefillResult(
                    globalHidden = extractLastHiddenTensor(outputs.requiredTensor("global_hidden")),
                    pastValidLengths = seqLen,
                    pastResult = outputs,
                )
            }
        }
    }

    private fun runDecode(prefillResult: PrefillResult, maxFrames: Int, seed: Long): List<IntArray> {
        val cfg = manifest.ttsConfig
        val audioTokens = ArrayList<IntArray>()
        val rowWidth = cfg.nVq + 1
        val cappedMaxFrames = maxFrames.coerceAtMost(manifest.generationDefaults.maxNewFrames)
        val previousTokenSets = Array(cfg.nVq) { HashSet<Int>() }
        val decodePastInputNames = ttsMeta.onnx.decodeInputNames.drop(2)
        val decodePresentOutputNames = ttsMeta.onnx.decodeOutputNames.drop(1)
        val random = java.util.Random(seed)
        var pastValidLengths = prefillResult.pastValidLengths
        var globalHidden = prefillResult.globalHidden
        var pastResult: OrtSession.Result? = prefillResult.pastResult

        try {
            for (step in 0 until cappedMaxFrames) {
                val frameResult = runLocalFixedSampledFrame(globalHidden, previousTokenSets, random)
                if (!frameResult.shouldContinue) {
                    break
                }
                val audioRow = IntArray(rowWidth) { index ->
                    if (index == 0) cfg.audioAssistantSlotTokenId else cfg.audioPadTokenId
                }
                for (quantizer in 0 until cfg.nVq) {
                    val token = frameResult.frame[quantizer]
                    audioRow[quantizer + 1] = token
                    previousTokenSets[quantizer].add(token)
                }
                audioTokens += frameResult.frame
                OnnxTensor.createTensor(
                    env,
                    IntBuffer.wrap(audioRow),
                    longArrayOf(1, 1, rowWidth.toLong()),
                ).use { inputTensor ->
                    OnnxTensor.createTensor(
                        env,
                        IntBuffer.wrap(intArrayOf(pastValidLengths)),
                        longArrayOf(1),
                    ).use { pastTensor ->
                        val feeds = linkedMapOf<String, OnnxTensorLike>(
                            "input_ids" to inputTensor,
                            "past_valid_lengths" to pastTensor,
                        )
                        val previousPastResult = pastResult ?: error("Missing decode KV cache")
                        for (index in decodePastInputNames.indices) {
                            feeds[decodePastInputNames[index]] =
                                previousPastResult.requiredTensor(decodePresentOutputNames[index])
                        }
                        val outputs = decodeSession.run(feeds)
                        val nextGlobalHidden = extractLastHiddenTensor(outputs.requiredTensor("global_hidden"))
                        globalHidden.close()
                        previousPastResult.close()
                        pastResult = outputs
                        globalHidden = nextGlobalHidden
                        pastValidLengths += 1
                    }
                }
            }
        } finally {
            globalHidden.close()
            pastResult?.close()
        }
        return audioTokens
    }

    private fun runLocalFixedSampledFrame(
        globalHidden: OnnxTensor,
        previousTokenSets: Array<HashSet<Int>>,
        random: java.util.Random,
    ): LocalFrameResult {
        val cfg = manifest.ttsConfig
        val audioCodebookSize = cfg.audioCodebookSizes.firstOrNull() ?: 1024
        val seenMask = IntArray(cfg.nVq * audioCodebookSize)
        for (channelIndex in previousTokenSets.indices) {
            val channelOffset = channelIndex * audioCodebookSize
            for (tokenId in previousTokenSets[channelIndex]) {
                if (tokenId in 0 until audioCodebookSize) {
                    seenMask[channelOffset + tokenId] = 1
                }
            }
        }
        val assistantRandom = floatArrayOf(random.nextDouble().coerceIn(1e-6, 1.0 - 1e-6).toFloat())
        val audioRandom = FloatArray(cfg.nVq) {
            random.nextDouble().coerceIn(1e-6, 1.0 - 1e-6).toFloat()
        }
        OnnxTensor.createTensor(
            env,
            IntBuffer.wrap(seenMask),
            longArrayOf(1, cfg.nVq.toLong(), audioCodebookSize.toLong()),
        ).use { seenTensor ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(assistantRandom), longArrayOf(1)).use { assistantTensor ->
                OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(audioRandom),
                    longArrayOf(1, cfg.nVq.toLong()),
                ).use { audioTensor ->
                    val outputs = localFixedFrameSession.run(
                        mapOf(
                            "global_hidden" to globalHidden,
                            "repetition_seen_mask" to seenTensor,
                            "assistant_random_u" to assistantTensor,
                            "audio_random_u" to audioTensor,
                        ),
                    )
                    outputs.use {
                        return LocalFrameResult(
                            shouldContinue = it.requiredTensor("should_continue").scalarInt() > 0,
                            frame = it.requiredTensor("frame_token_ids").intArrayValue(),
                        )
                    }
                }
            }
        }
    }

    private fun decodeAudioTokens(audioTokens: List<IntArray>): FloatArray {
        require(audioTokens.isNotEmpty()) { "No audio tokens generated" }
        val numFrames = audioTokens.size
        val numQuantizers = manifest.ttsConfig.nVq
        val audioCodesFlat = IntArray(numFrames * numQuantizers)
        var offset = 0
        for (frame in audioTokens) {
            for (quantizer in 0 until numQuantizers) {
                audioCodesFlat[offset++] = frame[quantizer]
            }
        }
        OnnxTensor.createTensor(
            env,
            IntBuffer.wrap(audioCodesFlat),
            longArrayOf(1, numFrames.toLong(), numQuantizers.toLong()),
        ).use { codesTensor ->
            OnnxTensor.createTensor(
                env,
                IntBuffer.wrap(intArrayOf(numFrames)),
                longArrayOf(1),
            ).use { lengthsTensor ->
                val outputs = codecDecodeSession.run(
                    mapOf(
                        "audio_codes" to codesTensor,
                        "audio_code_lengths" to lengthsTensor,
                    ),
                )
                outputs.use {
                    val audio = it.requiredTensor("audio").value as Array<*>
                    val batch = audio[0] as Array<*>
                    val channels = batch.map { channel -> channel as FloatArray }
                    val reportedLength = it.requiredTensor("audio_lengths").scalarInt()
                    val length = min(reportedLength, channels.minOfOrNull { channel -> channel.size } ?: 0)
                    return FloatArray(length) { sampleIndex ->
                        channels.sumOf { channel -> channel[sampleIndex].toDouble() }.toFloat() / channels.size
                    }
                }
            }
        }
    }

    private fun writeWavMono(audioData: FloatArray, sampleRate: Int, outputFile: File) {
        outputFile.parentFile?.mkdirs()
        val channels = 1
        val dataSize = audioData.size * 2
        val fileSize = 44 + dataSize
        val buffer = ByteBuffer.allocate(fileSize).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put("RIFF".toByteArray(Charsets.US_ASCII))
        buffer.putInt(fileSize - 8)
        buffer.put("WAVE".toByteArray(Charsets.US_ASCII))
        buffer.put("fmt ".toByteArray(Charsets.US_ASCII))
        buffer.putInt(16)
        buffer.putShort(1.toShort())
        buffer.putShort(channels.toShort())
        buffer.putInt(sampleRate)
        buffer.putInt(sampleRate * channels * 2)
        buffer.putShort((channels * 2).toShort())
        buffer.putShort(16.toShort())
        buffer.put("data".toByteArray(Charsets.US_ASCII))
        buffer.putInt(dataSize)
        for (sample in audioData) {
            buffer.putShort((sample.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        }
        outputFile.writeBytes(buffer.array())
    }

    companion object {
        private fun resolveManifestPath(modelRoot: File): File {
            val candidates = listOf(
                File(modelRoot, "browser_poc_manifest.json"),
                File(modelRoot, "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json"),
                File(modelRoot, "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json"),
            )
            return candidates.firstOrNull { it.isFile }
                ?: error("browser_poc_manifest.json not found. Tried: ${candidates.joinToString { it.absolutePath }}")
        }

        private fun readJson(file: File): JSONObject {
            require(file.isFile) { "Missing JSON file: ${file.absolutePath}" }
            return JSONObject(file.readText(Charsets.UTF_8))
        }

        private fun flattenIntTensorValue(raw: Any?): IntArray {
            val values = ArrayList<Int>()
            fun append(value: Any?) {
                when (value) {
                    is Int -> values += value
                    is Long -> values += value.toInt()
                    is Short -> values += value.toInt()
                    is Byte -> values += value.toInt()
                    is IntArray -> values += value.toList()
                    is LongArray -> value.forEach { values += it.toInt() }
                    is ShortArray -> value.forEach { values += it.toInt() }
                    is ByteArray -> value.forEach { values += it.toInt() }
                    is Array<*> -> value.forEach { append(it) }
                    null -> Unit
                    else -> error("Unsupported int tensor value: ${value.javaClass}")
                }
            }
            append(raw)
            return values.toIntArray()
        }

        private fun extractLastHiddenTensor(tensor: OnnxTensor): OnnxTensor {
            val shape = tensor.info.shape
            val hidden = when (shape.size) {
                2 -> {
                    val value = tensor.value as Array<*>
                    value[0] as FloatArray
                }
                3 -> {
                    val value = tensor.value as Array<*>
                    val batch = value[0] as Array<*>
                    batch[batch.size - 1] as FloatArray
                }
                else -> error("Unexpected global_hidden rank: ${shape.size}")
            }
            return OnnxTensor.createTensor(
                OrtEnvironment.getEnvironment(),
                FloatBuffer.wrap(hidden.copyOf()),
                longArrayOf(1, hidden.size.toLong()),
            )
        }

        private fun OrtSession.Result.requiredValue(name: String): OnnxValue {
            return get(name).orElseThrow { IllegalStateException("Missing ONNX output: $name") }
        }

        private fun OrtSession.Result.requiredTensor(name: String): OnnxTensor {
            return requiredValue(name) as OnnxTensor
        }

        private fun OnnxTensor.scalarInt(): Int {
            return flattenIntTensorValue(value).firstOrNull() ?: error("Scalar int tensor is empty")
        }

        private fun OnnxTensor.intArrayValue(): IntArray {
            return flattenIntTensorValue(value)
        }
    }

    private data class InputRows(val inputIds: Array<IntArray>, val attentionMask: IntArray)
    private data class PrefillResult(
        val globalHidden: OnnxTensor,
        val pastValidLengths: Int,
        val pastResult: OrtSession.Result,
    )
    private data class LocalFrameResult(val shouldContinue: Boolean, val frame: IntArray)
}

data class SynthesisResult(
    val outputFile: File,
    val generatedFrames: Int,
    val sampleRate: Int,
    val durationMs: Long,
    val elapsedMs: Long,
)

private data class ModelManifest(
    val modelFiles: ModelFiles,
    val ttsConfig: TtsConfig,
    val promptTemplates: PromptTemplates,
    val generationDefaults: GenerationDefaults,
    val builtinVoices: List<BuiltinVoice>,
) {
    companion object {
        fun fromJson(json: JSONObject): ModelManifest {
            return ModelManifest(
                modelFiles = ModelFiles.fromJson(json.getJSONObject("model_files")),
                ttsConfig = TtsConfig.fromJson(json.getJSONObject("tts_config")),
                promptTemplates = PromptTemplates.fromJson(json.getJSONObject("prompt_templates")),
                generationDefaults = GenerationDefaults.fromJson(json.optJSONObject("generation_defaults")),
                builtinVoices = json.optJSONArray("builtin_voices")?.let { voices ->
                    List(voices.length()) { index -> BuiltinVoice.fromJson(voices.getJSONObject(index)) }
                } ?: emptyList(),
            )
        }
    }
}

private data class ModelFiles(
    val ttsMeta: String,
    val codecMeta: String,
) {
    companion object {
        fun fromJson(json: JSONObject): ModelFiles {
            return ModelFiles(
                ttsMeta = json.getString("tts_meta"),
                codecMeta = json.getString("codec_meta"),
            )
        }
    }
}

private data class TtsConfig(
    val nVq: Int,
    val audioPadTokenId: Int,
    val audioStartTokenId: Int,
    val audioEndTokenId: Int,
    val audioUserSlotTokenId: Int,
    val audioAssistantSlotTokenId: Int,
    val audioCodebookSizes: IntArray,
) {
    companion object {
        fun fromJson(json: JSONObject): TtsConfig {
            return TtsConfig(
                nVq = json.getInt("n_vq"),
                audioPadTokenId = json.getInt("audio_pad_token_id"),
                audioStartTokenId = json.getInt("audio_start_token_id"),
                audioEndTokenId = json.getInt("audio_end_token_id"),
                audioUserSlotTokenId = json.optInt("audio_user_slot_token_id", 8),
                audioAssistantSlotTokenId = json.getInt("audio_assistant_slot_token_id"),
                audioCodebookSizes = json.getJSONArray("audio_codebook_sizes").toIntArrayCompat(),
            )
        }

        private fun JSONArray.toIntArrayCompat(): IntArray {
            return IntArray(length()) { index -> getInt(index) }
        }
    }
}

private data class PromptTemplates(
    val userPromptPrefixTokenIds: IntArray,
    val userPromptAfterReferenceTokenIds: IntArray,
    val assistantPromptPrefixTokenIds: IntArray,
) {
    companion object {
        fun fromJson(json: JSONObject): PromptTemplates {
            return PromptTemplates(
                userPromptPrefixTokenIds = json.getJSONArray("user_prompt_prefix_token_ids").toIntArrayCompat(),
                userPromptAfterReferenceTokenIds =
                    json.getJSONArray("user_prompt_after_reference_token_ids").toIntArrayCompat(),
                assistantPromptPrefixTokenIds =
                    json.getJSONArray("assistant_prompt_prefix_token_ids").toIntArrayCompat(),
            )
        }

        private fun JSONArray.toIntArrayCompat(): IntArray {
            return IntArray(length()) { index -> getInt(index) }
        }
    }
}

private data class GenerationDefaults(val maxNewFrames: Int = 375) {
    companion object {
        fun fromJson(json: JSONObject?): GenerationDefaults {
            return GenerationDefaults(maxNewFrames = json?.optInt("max_new_frames", 375) ?: 375)
        }
    }
}

private data class BuiltinVoice(
    val voice: String,
    val promptAudioCodes: List<IntArray>,
) {
    companion object {
        fun fromJson(json: JSONObject): BuiltinVoice {
            return BuiltinVoice(
                voice = json.optString("voice", ""),
                promptAudioCodes = json.optJSONArray("prompt_audio_codes")?.let { outer ->
                    List(outer.length()) { index ->
                        val row = outer.getJSONArray(index)
                        IntArray(row.length()) { itemIndex -> row.getInt(itemIndex) }
                    }
                } ?: emptyList(),
            )
        }
    }
}

private data class TtsMeta(
    val files: TtsFiles,
    val onnx: TtsOnnxNames,
) {
    companion object {
        fun fromJson(json: JSONObject): TtsMeta {
            return TtsMeta(
                files = TtsFiles.fromJson(json.getJSONObject("files")),
                onnx = TtsOnnxNames.fromJson(json.getJSONObject("onnx")),
            )
        }
    }
}

private data class TtsFiles(
    val prefill: String,
    val decodeStep: String,
    val localFixedSampledFrame: String,
) {
    companion object {
        fun fromJson(json: JSONObject): TtsFiles {
            return TtsFiles(
                prefill = json.getString("prefill"),
                decodeStep = json.getString("decode_step"),
                localFixedSampledFrame = json.getString("local_fixed_sampled_frame"),
            )
        }
    }
}

private data class TtsOnnxNames(
    val decodeInputNames: List<String>,
    val decodeOutputNames: List<String>,
) {
    companion object {
        fun fromJson(json: JSONObject): TtsOnnxNames {
            return TtsOnnxNames(
                decodeInputNames = json.getJSONArray("decode_input_names").toStringList(),
                decodeOutputNames = json.getJSONArray("decode_output_names").toStringList(),
            )
        }

        private fun JSONArray.toStringList(): List<String> {
            return List(length()) { index -> getString(index) }
        }
    }
}

private data class CodecMeta(
    val files: CodecFiles,
    val codecConfig: CodecConfig,
) {
    companion object {
        fun fromJson(json: JSONObject): CodecMeta {
            return CodecMeta(
                files = CodecFiles.fromJson(json.getJSONObject("files")),
                codecConfig = CodecConfig.fromJson(json.getJSONObject("codec_config")),
            )
        }
    }
}

private data class CodecFiles(val decodeFull: String) {
    companion object {
        fun fromJson(json: JSONObject): CodecFiles {
            return CodecFiles(decodeFull = json.getString("decode_full"))
        }
    }
}

private data class CodecConfig(val sampleRate: Int) {
    companion object {
        fun fromJson(json: JSONObject): CodecConfig {
            return CodecConfig(sampleRate = json.getInt("sample_rate"))
        }
    }
}
