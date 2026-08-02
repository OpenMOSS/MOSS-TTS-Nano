package com.openmoss.ttsnano.onnxruntime

import android.app.Activity
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File

class MainActivity : Activity() {
    private val mainHandler = Handler(Looper.getMainLooper())
    private lateinit var logView: TextView
    private lateinit var customTextInput: EditText
    private lateinit var generateCustomButton: Button
    private lateinit var generateEnglishButton: Button
    private lateinit var generateChineseButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = getString(R.string.app_name)

        logView = TextView(this).apply {
            textSize = 14f
            setTextIsSelectable(true)
        }
        customTextInput = EditText(this).apply {
            setText("Hello world!")
            hint = "Custom text"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            minLines = 2
        }
        generateCustomButton = Button(this).apply {
            text = "Generate custom text WAV"
            setOnClickListener {
                runCustomText(customTextInput.text.toString())
            }
        }
        generateEnglishButton = Button(this).apply {
            text = "Generate English demo WAV"
            setOnClickListener {
                runDemo("en", DemoPrompts.ENGLISH_TOKEN_IDS)
            }
        }
        generateChineseButton = Button(this).apply {
            text = "Generate Chinese demo WAV"
            setOnClickListener {
                runDemo("zh", DemoPrompts.CHINESE_TOKEN_IDS)
            }
        }

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
            addView(customTextInput)
            addView(generateCustomButton)
            addView(generateEnglishButton)
            addView(generateChineseButton)
            addView(
                logView,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
        }
        setContentView(ScrollView(this).apply { addView(content) })
        appendLog("Place model files under:\n${modelRoot().absolutePath}")
        appendLog("Enter text or tap a pre-tokenized demo prompt.")
    }

    private fun runCustomText(text: String) {
        val trimmedText = text.trim()
        if (trimmedText.isEmpty()) {
            appendLog("[custom] text is empty")
            return
        }
        setButtonsEnabled(false)
        appendLog("\n[custom] starting synthesis: $trimmedText")
        Thread {
            try {
                val outputFile = File(cacheDir, "moss_tts_nano_android_custom.wav")
                MossOnnxDemoEngine(
                    modelRoot = modelRoot(),
                    outputDir = cacheDir,
                    cpuThreads = 2,
                ).use { engine ->
                    val result = engine.synthesizeText(
                        text = trimmedText,
                        outputFile = outputFile,
                        voice = "Junhao",
                        maxFrames = 160,
                        seed = 1234L,
                    )
                    appendLogFromWorker(
                        "[custom] done: ${result.outputFile.absolutePath}\n" +
                            "frames=${result.generatedFrames} " +
                            "sampleRate=${result.sampleRate}Hz " +
                            "durationMs=${result.durationMs} " +
                            "elapsedMs=${result.elapsedMs}",
                    )
                }
            } catch (error: Throwable) {
                appendLogFromWorker("[custom] failed: ${error.javaClass.simpleName}: ${error.message}")
            } finally {
                mainHandler.post { setButtonsEnabled(true) }
            }
        }.start()
    }

    private fun runDemo(label: String, textTokenIds: IntArray) {
        setButtonsEnabled(false)
        appendLog("\n[$label] starting synthesis")
        Thread {
            try {
                val outputFile = File(cacheDir, "moss_tts_nano_android_$label.wav")
                MossOnnxDemoEngine(
                    modelRoot = modelRoot(),
                    outputDir = cacheDir,
                    cpuThreads = 2,
                ).use { engine ->
                    val result = engine.synthesize(
                        textTokenIds = textTokenIds,
                        outputFile = outputFile,
                        voice = "Junhao",
                        maxFrames = 160,
                        seed = 1234L,
                    )
                    appendLogFromWorker(
                        "[$label] done: ${result.outputFile.absolutePath}\n" +
                            "frames=${result.generatedFrames} " +
                            "sampleRate=${result.sampleRate}Hz " +
                            "durationMs=${result.durationMs} " +
                            "elapsedMs=${result.elapsedMs}",
                    )
                }
            } catch (error: Throwable) {
                appendLogFromWorker("[$label] failed: ${error.javaClass.simpleName}: ${error.message}")
            } finally {
                mainHandler.post { setButtonsEnabled(true) }
            }
        }.start()
    }

    private fun modelRoot(): File {
        return File(getExternalFilesDir(null), "moss_tts_onnx").apply { mkdirs() }
    }

    private fun setButtonsEnabled(enabled: Boolean) {
        generateCustomButton.isEnabled = enabled
        generateEnglishButton.isEnabled = enabled
        generateChineseButton.isEnabled = enabled
    }

    private fun appendLog(message: String) {
        logView.append(message + "\n")
    }

    private fun appendLogFromWorker(message: String) {
        mainHandler.post { appendLog(message) }
    }
}
