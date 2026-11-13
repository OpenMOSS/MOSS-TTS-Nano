import unittest

import numpy as np

from onnx_tts_runtime import OnnxTtsRuntime, ReadingChunk, _join_reading_audio, _PARAGRAPH_BREAK


class CharacterCountRuntime(OnnxTtsRuntime):
    def __init__(self):
        pass

    def count_text_tokens(self, text: str) -> int:
        return len(text)


class AutomaticProsodyTest(unittest.TestCase):
    def test_keeps_contrast_in_one_chunk_and_preserves_paragraph_break(self):
        contrast = "交稿时间不是周三，是周五。"
        paragraphs = [f"请确认。{contrast}", "我今天才知道。"]
        chunks = CharacterCountRuntime().plan_reading_chunks(paragraphs, 75)
        self.assertEqual([chunk.text for chunk in chunks], ["请确认。", contrast, paragraphs[1]])
        self.assertEqual([chunk.pause_ms for chunk in chunks], [280, 460, 460])
        self.assertIsNotNone(chunks[1].focus_fraction)
        self.assertIsNone(chunks[2].focus_fraction)

    def test_changes_clause_gain_only_when_a_matching_pause_is_available(self):
        chunk = ReadingChunk("不是周三，是周五。", 280, 0.5)
        speech = np.full((400, 2), 0.5, dtype=np.float32)
        audio = np.concatenate((speech, np.zeros((100, 2), dtype=np.float32), speech))
        adjusted = _join_reading_audio([chunk], [audio], 1000)
        self.assertEqual(adjusted.shape, audio.shape)
        np.testing.assert_allclose(adjusted[:400], 0.46)
        np.testing.assert_allclose(adjusted[-400:], 0.60)
        continuous = np.tile(speech, (2, 1))
        np.testing.assert_array_equal(_join_reading_audio([chunk], [continuous], 1000), continuous)

    def test_adjusts_existing_silence_instead_of_adding_a_full_pause(self):
        speech = np.full((100, 2), 0.5, dtype=np.float32)
        chunks = [ReadingChunk("短句，", 150), ReadingChunk("下一句。", 280)]
        for silence_length in (60, 200):
            silence = np.zeros((silence_length, 2), dtype=np.float32)
            result = _join_reading_audio(
                chunks, [np.concatenate((speech, silence)), np.concatenate((silence, speech))], 1000
            )
            np.testing.assert_array_equal(result[:100], speech)
            np.testing.assert_array_equal(result[-100:], speech)
            np.testing.assert_array_equal(result[100:-100], np.zeros((190, 2)))

    def test_handles_1d_mono_audio_and_preserves_shape(self):
        chunk = ReadingChunk("不是周三，是周五。", 280, 0.5)
        speech = np.full(400, 0.5, dtype=np.float32)
        audio = np.concatenate((speech, np.zeros(100, dtype=np.float32), speech))
        adjusted = _join_reading_audio([chunk], [audio], 1000)
        self.assertEqual(adjusted.ndim, 1)
        self.assertEqual(adjusted.shape, audio.shape)
        np.testing.assert_allclose(adjusted[:400], 0.46)
        np.testing.assert_allclose(adjusted[-400:], 0.60)

    def test_handles_silent_audio_chunk(self):
        chunks = [ReadingChunk("短句，", 150), ReadingChunk("下一句。", 280)]
        silence1 = np.zeros((100, 2), dtype=np.float32)
        silence2 = np.zeros((100, 2), dtype=np.float32)
        result = _join_reading_audio(chunks, [silence1, silence2], 1000)
        self.assertEqual(result.shape, (250, 2))
        self.assertTrue(np.all(result == 0))

    def test_recognizes_contrast_with_digits_english_and_spaces(self):
        runtime = CharacterCountRuntime()
        chunks = runtime.plan_reading_chunks(
            ["请确认。 交稿时间不是3号，是5号。", "“虽然价格是100元，但是质量很好。”"], 75
        )
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["请确认。", "交稿时间不是3号，是5号。", "“虽然价格是100元，但是质量很好。”"],
        )
        self.assertIsNotNone(chunks[1].focus_fraction)
        self.assertIsNotNone(chunks[2].focus_fraction)

    def test_paragraph_break_regex_with_consecutive_blank_lines(self):
        text = "段落一\n\n\n段落二\r\n\r\n\r\n段落三"
        splits = [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]
        self.assertEqual(splits, ["段落一", "段落二", "段落三"])


if __name__ == "__main__":
    unittest.main()
