"""Core tests with a mock provider - no API calls, no model downloads."""

import json

import pytest

from subkit.chunker import build_context_window, chunk_subtitles
from subkit.config import AppConfig
from subkit.engine import TranslationEngine, translate_document_sync
from subkit.jellyfin import categorize_item, plan_library, sidecar_path, write_sidecar_atomic
from subkit.merger import format_srt
from subkit.parser import Subtitle, SubtitleDocument, parse_srt_content

SRT = """1
00:00:01,000 --> 00:00:03,000
Hello there.

2
00:00:03,500 --> 00:00:05,000
How are you?

3
00:00:10,000 --> 00:00:12,000
[Music]

4
00:00:12,500 --> 00:00:14,500
The hyperdrive is broken, captain.
"""


def make_doc(n=4):
    return SubtitleDocument(format="srt", subtitles=parse_srt_content(SRT))


class MockProvider:
    """Returns FI:<text> for every id; echoes a fixed new glossary term."""

    name = "mock"

    def __init__(self, drop_ids=(), new_terms=None, garbage_first=False):
        self.drop_ids = set(drop_ids)
        self.new_terms = new_terms or {}
        self.garbage_first = garbage_first
        self.calls = []
        self._last_payload = []

    async def chat(self, system, messages):
        self.calls.append({"system": system, "messages": messages})
        if self.garbage_first and len(self.calls) == 1:
            return "sure, here you go"  # unparseable -> correction path
        user = messages[-1]["content"]
        if "SEGMENTS:\n" not in user:
            # correction retry: re-emit from the payload we last saw
            payload = self._last_payload
        else:
            payload = json.loads(user.split("SEGMENTS:\n", 1)[1])
            self._last_payload = payload
        segs = [
            {"id": p["id"], "text": "FI:" + p["text"]}
            for p in payload
            if p["id"] not in self.drop_ids
        ]
        return (
            "<glossary>" + json.dumps(self.new_terms) + "</glossary>\n"
            "<translation>" + json.dumps({"segments": segs}) + "</translation>"
        )


def cfg(**kw):
    c = AppConfig()
    c.translator.target_language = "Finnish"
    for k, v in kw.items():
        setattr(c.translator, k, v)
    return c


def test_parser_roundtrip():
    subs = parse_srt_content(SRT)
    assert len(subs) == 4
    assert subs[0].start_ms == 1000 and subs[0].end_ms == 3000
    assert subs[3].text == "The hyperdrive is broken, captain."


def test_chunker_splits_on_time_gap():
    doc = make_doc()
    chunks = chunk_subtitles(doc.subtitles, AppConfig().chunk)
    # gap of 5s before cue 3 -> at least 2 groups
    assert len(chunks) >= 2
    assert sum(len(c) for c in chunks) == 4


def test_context_window_is_previous_chunk_tail():
    doc = make_doc()
    c = AppConfig()
    chunks = chunk_subtitles(doc.subtitles, c.chunk)
    ctx = build_context_window(chunks, 2)
    assert ctx[0] is None
    for i in range(1, len(chunks)):
        assert ctx[i] == chunks[i - 1][-2:]


def test_translate_full():
    report = translate_document_sync(make_doc(), cfg(), MockProvider())
    assert report.ok
    assert all(s.text.startswith("FI:") or s.text.startswith("[") for s in report.subtitles)
    assert report.chunks_done == report.chunks_total


def test_dropped_id_falls_back_and_is_flagged():
    doc = make_doc()
    dropped = doc.subtitles[1].id
    report = translate_document_sync(make_doc(), cfg(), MockProvider(drop_ids={dropped}))
    assert dropped in report.dropped_ids
    assert not report.ok
    sub = next(s for s in report.subtitles if s.id == dropped)
    assert sub.text == doc.subtitles[1].text  # original kept


def test_rolling_glossary_carried_between_chunks():
    c = cfg()
    c.chunk.max_lines = 1  # force one cue per chunk
    provider = MockProvider(new_terms={"hyperdrive": "hyperajuri"})
    report = translate_document_sync(make_doc(), c, provider)
    assert report.glossary.get("hyperdrive") == "hyperajuri"
    # a later call's user message must contain the carried glossary term
    later = [c for c in provider.calls[1:] if "hyperajuri" in c["messages"][-1]["content"]]
    assert later, "rolling glossary term was not carried into a later chunk"


def test_correction_retry_on_garbage():
    provider = MockProvider(garbage_first=True)
    report = translate_document_sync(make_doc(), cfg(), provider)
    assert report.chunks_done >= 1
    assert len(provider.calls) >= 2  # original + correction


def test_payload_has_cps_budget():
    provider = MockProvider()
    translate_document_sync(make_doc(), cfg(), provider)
    user = provider.calls[0]["messages"][-1]["content"]
    payload = json.loads(user.split("SEGMENTS:\n", 1)[1])
    assert all("max_chars" in p for p in payload)
    assert payload[0]["max_chars"] >= 20


def test_gemini_request_shape(monkeypatch):
    from subkit.config import ProviderConfig
    from subkit.providers import GeminiProvider

    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

    class FakeClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["body"] = json
            captured["headers"] = headers
            return FakeResp()

    monkeypatch.setattr("subkit.providers.httpx.AsyncClient", FakeClient)
    p = GeminiProvider(ProviderConfig(kind="gemini", model="gemini-3.8-flash", api_key="k"))
    import asyncio

    out = asyncio.run(p.chat("SYS", [{"role": "user", "content": "hi"}]))
    assert out == "ok"
    assert captured["url"].endswith("/models/gemini-3.8-flash:generateContent")
    assert captured["headers"]["x-goog-api-key"] == "k"
    assert captured["body"]["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert captured["body"]["contents"][0]["role"] == "user"
    assert captured["body"]["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"


def test_categorize_matrix(tmp_path, monkeypatch):
    def no_embed(v):
        return set()

    monkeypatch.setattr("subkit.jellyfin.embedded_sub_langs", no_embed)
    v = tmp_path / "Movie (2020).mkv"
    v.write_bytes(b"")
    # no subs at all -> transcribe
    assert categorize_item(v, "fi") == "transcribe"
    # English sidecar -> translate from it (English subs must NOT cause a skip)
    (tmp_path / "Movie (2020).en.srt").write_text("x", encoding="utf-8")
    assert categorize_item(v, "fi") == "translate"
    # embedded English also a translate source
    (tmp_path / "Movie (2020).en.srt").unlink()
    monkeypatch.setattr("subkit.jellyfin.embedded_sub_langs", lambda v: {"eng"})
    assert categorize_item(v, "fi") == "translate"
    # Finnish embedded -> skip
    monkeypatch.setattr("subkit.jellyfin.embedded_sub_langs", lambda v: {"fin"})
    assert categorize_item(v, "fi") == "skip"
    monkeypatch.setattr("subkit.jellyfin.embedded_sub_langs", no_embed)
    # user-owned Finnish sidecar -> skip, and --force must NOT regenerate over it
    (tmp_path / "Movie (2020).fi.srt").write_text("x", encoding="utf-8")
    assert categorize_item(v, "fi") == "skip"
    assert categorize_item(v, "fi", force=True) == "skip"
    # our own generated file -> skip normally, re-done under --force
    (tmp_path / "Movie (2020).fi.srt").unlink()
    (tmp_path / "Movie (2020).fi.generated.srt").write_text("x", encoding="utf-8")
    assert categorize_item(v, "fi") == "skip"
    assert categorize_item(v, "fi", force=True) == "transcribe"
    assert sidecar_path(v, "fi").name == "Movie (2020).fi.generated.srt"


def test_sidecar_write_protection(tmp_path):
    video = tmp_path / "Film.mkv"
    video.write_bytes(b"")
    out = write_sidecar_atomic(video, "fi", "1\n00:00:01,000 --> 00:00:02,000\nHei\n")
    assert out.exists() and out.name == "Film.fi.generated.srt"
    # no force: never overwrite, even our own file
    with pytest.raises(FileExistsError):
        write_sidecar_atomic(video, "fi", "other")
    # force: our generated file may be regenerated
    out2 = write_sidecar_atomic(video, "fi", "new", force=True)
    assert out2.read_text(encoding="utf-8") == "new"
    # force must never overwrite a user-owned (non-generated) subtitle
    user_file = tmp_path / "Film.fi.srt"
    user_file.write_text("mine", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_sidecar_atomic(video, "fi", "other", generated_marker=False, force=True)
    assert user_file.read_text(encoding="utf-8") == "mine"


def test_plan_library(tmp_path, monkeypatch):
    (tmp_path / "a.mkv").write_bytes(b"")  # nothing -> transcribe
    (tmp_path / "b.mkv").write_bytes(b"")
    (tmp_path / "b.en.srt").write_text("x", encoding="utf-8")  # english -> translate
    (tmp_path / "c.mkv").write_bytes(b"")
    (tmp_path / "c.fi.srt").write_text("x", encoding="utf-8")  # finnish -> skip
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr("subkit.jellyfin.embedded_sub_langs", lambda v: set())
    plan = plan_library(tmp_path, "fi")
    assert [p.name for p in plan.transcribe] == ["a.mkv"]
    assert [p.name for p in plan.translate] == ["b.mkv"]
    assert [p.name for p in plan.skipped] == ["c.mkv"]


def test_full_transcript_context_sent():
    provider = MockProvider()
    doc = make_doc()
    translate_document_sync(doc, cfg(), provider)
    user = provider.calls[0]["messages"][-1]["content"]
    assert "FULL TRANSCRIPT (read-only" in user
    # every original cue appears in the full-transcript block
    assert "hyperdrive" in user and "How are you?" in user


def test_full_context_can_be_disabled():
    provider = MockProvider()
    c = cfg(full_context=False)
    translate_document_sync(make_doc(), c, provider)
    user = provider.calls[0]["messages"][-1]["content"]
    assert "FULL TRANSCRIPT" not in user


def test_request_budget_stops_run():
    provider = MockProvider()
    c = cfg(max_requests=1)
    c.chunk.max_lines = 1  # 3 translatable cues -> 3 chunks, budget allows 1 call
    with pytest.raises(RuntimeError, match="budget"):
        translate_document_sync(make_doc(), c, provider)


def test_resume_skips_done_chunks(tmp_path):
    provider = MockProvider()
    c = cfg(status_dir=str(tmp_path))
    c.chunk.max_lines = 2
    doc = make_doc()
    rid = str(tmp_path / "movie.srt")
    import asyncio

    asyncio.run(TranslationEngine(c, provider).translate_document(doc, resume_id=rid))
    assert (tmp_path / "movie.srt.subkit-status.json").exists()
    provider2 = MockProvider()
    report2 = asyncio.run(TranslationEngine(c, provider2).translate_document(doc, resume_id=rid))
    assert provider2.calls == [], "resume should make no provider calls when all chunks done"
    assert report2.chunks_done == report2.chunks_total


def test_resume_invalidated_on_language_change(tmp_path):
    import asyncio

    c = cfg(status_dir=str(tmp_path))
    rid = str(tmp_path / "m.srt")
    p1 = MockProvider()
    asyncio.run(TranslationEngine(c, p1).translate_document(make_doc(), resume_id=rid))
    c2 = cfg(status_dir=str(tmp_path))
    c2.translator.target_language = "Swedish"  # config changed
    p2 = MockProvider()
    asyncio.run(TranslationEngine(c2, p2).translate_document(make_doc(), resume_id=rid))
    assert p2.calls, "stale resume reused after target language change"


def test_resume_invalidated_on_source_change(tmp_path):
    import asyncio

    c = cfg(status_dir=str(tmp_path))
    rid = str(tmp_path / "m.srt")
    p1 = MockProvider()
    asyncio.run(TranslationEngine(c, p1).translate_document(make_doc(), resume_id=rid))
    doc2 = make_doc()
    doc2.subtitles[0].text = "Changed line."  # source text changed
    p2 = MockProvider()
    asyncio.run(TranslationEngine(c, p2).translate_document(doc2, resume_id=rid))
    assert p2.calls, "stale resume reused after source text change"


def test_resume_invalidated_on_model_change(tmp_path):
    import asyncio

    c = cfg(status_dir=str(tmp_path))
    rid = str(tmp_path / "m.srt")
    asyncio.run(TranslationEngine(c, MockProvider()).translate_document(make_doc(), resume_id=rid))
    c2 = cfg(status_dir=str(tmp_path))
    c2.provider.model = "other-model"
    p2 = MockProvider()
    asyncio.run(TranslationEngine(c2, p2).translate_document(make_doc(), resume_id=rid))
    assert p2.calls, "stale resume reused after model change"


def test_budget_counts_correction_calls(tmp_path):
    from subkit.engine import BudgetExhausted

    provider = MockProvider(garbage_first=True)  # needs original + correction call
    c = cfg(max_requests=1)
    with pytest.raises(BudgetExhausted):
        translate_document_sync(make_doc(), c, provider)
    assert len(provider.calls) == 1  # stopped before spending the correction call...
    # budget allowed 1 call; correction would be call 2 -> blocked


def test_budget_counts_compression_calls():
    from subkit.engine import BudgetExhausted

    class WordyProvider(MockProvider):
        async def chat(self, system, messages):
            out = await super().chat(system, messages)
            # make every translation far over the CPS budget
            return out.replace("FI:", "FI:" + "word " * 40)

    provider = WordyProvider()
    c = cfg(max_requests=1, cps_target=5.0, cps_min_chars=5, cps_tolerance=1.0)
    with pytest.raises(BudgetExhausted):
        translate_document_sync(make_doc(), c, provider)


def test_context_budget_fails_before_sending():
    provider = MockProvider()
    c = cfg(max_context_chars=10)  # any real transcript exceeds this
    with pytest.raises(RuntimeError, match="MAX_CONTEXT_CHARS"):
        translate_document_sync(make_doc(), c, provider)
    assert provider.calls == [], "provider must not be called when over context budget"


def test_gemini_thinking_auto_omit_on_non3x(monkeypatch):
    from subkit.config import ProviderConfig
    from subkit.providers import GeminiProvider

    monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)
    p = GeminiProvider(
        ProviderConfig(kind="gemini", model="gemini-2.5-flash", api_key="k", thinking_level="low")
    )
    assert p._cfg.thinking_level == "off", "default thinking must auto-omit on non-3.x"


def test_gemini_thinking_explicit_on_non3x_is_clean_error(monkeypatch):
    from subkit.config import ProviderConfig
    from subkit.providers import GeminiProvider

    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "low")
    with pytest.raises(ValueError) as ei:
        GeminiProvider(
            ProviderConfig(
                kind="gemini", model="gemini-2.5-flash", api_key="k", thinking_level="low"
            )
        )
    assert "GEMINI_THINKING_LEVEL=off" in str(ei.value)


def test_format_srt_empty_list():
    assert format_srt([]) == ""


def test_format_srt_single_subtitle():
    subs = [Subtitle(id=1, start="00:00:01,000", end="00:00:03,000", text="Hello there.")]
    assert format_srt(subs) == "1\n00:00:01,000 --> 00:00:03,000\nHello there.\n"


def test_format_srt_multiline_text_is_collapsed():
    subs = [
        Subtitle(
            id=2,
            start="00:00:03,500",
            end="00:00:05,000",
            text="Line one\nLine two",
        )
    ]
    assert format_srt(subs) == "2\n00:00:03,500 --> 00:00:05,000\nLine one Line two\n"


def test_provider_client_reuse_and_close(monkeypatch):
    from subkit.config import ProviderConfig
    from subkit.providers import GeminiProvider, OpenAIProvider

    class DummyClient:
        def __init__(self, **kwargs):
            self.is_closed = False

        async def aclose(self):
            self.is_closed = True

        async def post(self, url, json=None, headers=None):
            class Resp:
                status_code = 200

                def json(self):
                    return {
                        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                        "choices": [{"message": {"content": "ok"}}],
                    }

            return Resp()

    monkeypatch.setattr("subkit.providers.httpx.AsyncClient", DummyClient)

    import asyncio

    async def run():
        p1 = GeminiProvider(ProviderConfig(kind="gemini", model="gemini-3.8-flash", api_key="k"))
        c1_a = p1._get_client()
        c1_b = p1._get_client()
        assert c1_a is c1_b
        await p1.close()
        assert c1_a.is_closed

        p2 = OpenAIProvider(ProviderConfig(kind="openai", model="gpt-4o", api_key="k"))
        c2_a = p2._get_client()
        c2_b = p2._get_client()
        assert c2_a is c2_b
        await p2.close()
        assert c2_a.is_closed

    asyncio.run(run())


def test_ass_tag_preservation_in_translation():
    ass_sub = Subtitle(
        id=1,
        start="00:00:01.00",
        end="00:00:03.00",
        text="{\\pos(100,200)}{\\i1}Hello world{\\i0}",
    )
    doc = SubtitleDocument(format="ass", subtitles=[ass_sub], ass_lines=["Dialogue: ..."])

    class AssMockProvider(MockProvider):
        async def chat(self, system, messages):
            user = messages[-1]["content"]
            payload = json.loads(user.split("SEGMENTS:\n", 1)[1])
            # Check that ASS tags were stripped before sending to LLM
            assert "{\\pos" not in payload[0]["text"]
            assert "{\\i1}" not in payload[0]["text"]
            assert payload[0]["text"] == "Hello world"
            return (
                "<glossary>{}</glossary>\n"
                '<translation>{"segments": [{"id": 1, "text": "Hei maailma"}]}</translation>'
            )

    report = translate_document_sync(doc, cfg(), AssMockProvider())
    assert report.ok
    res_text = report.subtitles[0].text
    # Check that tags were restored after translation
    assert res_text.startswith("{\\pos(100,200)}")
    assert "{\\i1}" in res_text and "{\\i0}" in res_text


def test_loads_object_markdown_fences():
    from subkit.engine import _loads_object

    block = '```json\n{"segments": [{"id": 1, "text": "hi"}]}\n```'
    res = _loads_object(block)
    assert res == {"segments": [{"id": 1, "text": "hi"}]}
