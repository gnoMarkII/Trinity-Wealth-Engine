from tools.youtube_tools import _extract_video_id


class TestExtractVideoID:
    def test_youtube_com_watch(self):
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_youtu_be_short(self):
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed(self):
        assert _extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts(self):
        assert _extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_live(self):
        assert _extract_video_id("https://www.youtube.com/live/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_raw_id(self):
        assert _extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_id_with_dashes_and_underscores(self):
        # 11 chars exactly: a(1)-(2)1(3)_(4)b(5)2(6)-(7)c(8)_(9)d(10)3(11)
        assert _extract_video_id("a-1_b2-c_d3") == "a-1_b2-c_d3"

    def test_invalid_returns_none(self):
        assert _extract_video_id("not a youtube url") is None
        assert _extract_video_id("") is None

    def test_url_with_extra_params(self):
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s") == "dQw4w9WgXcQ"
