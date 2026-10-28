"""Persistent per-series glossary: provenance, isolation, engine wiring."""

import json

import pytest
from test_core import MockProvider, cfg, make_doc

from subkit.engine import TranslationEngine, _fingerprint, translate_document_sync
from subkit.series import SeriesGlossary, list_series, series_path


def test_roundtrip_with_provenance(tmp_path):
    path = tmp_path / "LotR.json"
    g = SeriesGlossary(name="LotR")
    g.set_user("Hobbit", "hobitti")
    g.merge_auto({"Shire": "Kontu"})
    g.save(path)
    back = SeriesGlossary.load(path)
    assert back.entries() == {"Hobbit": "hobitti", "Shire": "Kontu"}
    assert back.terms["Hobbit"]["origin"] == "user"
    assert back.terms["Shire"]["origin"] == "auto"


def test_merge_auto_never_overwrites_user(tmp_path):
    g = SeriesGlossary(name="LotR")
    g.set_user("Hobbit", "hobitti")
    added = g.merge_auto({"Hobbit": "haltija"})
    assert added == 0
    assert g.terms["Hobbit"]["target"] == "hobitti"
    assert g.terms["Hobbit"]["origin"] == "user"


def test_merge_auto_conflict_keeps_first_candidate():
    g = SeriesGlossary(name="LotR")
    assert g.merge_auto({"Ring": "Sormus"}) == 1
    assert g.merge_auto({"Ring": "Rengas"}) == 0  # conflicting candidate kept out
    assert g.terms["Ring"]["target"] == "Sormus"
    assert g.merge_auto({"Ring": "Sormus"}) == 0  # same rendering bumps count only
    assert g.terms["Ring"]["count"] == 2


def test_confirm_promotes_candidate_only():
    g = SeriesGlossary(name="LotR")
    g.merge_auto({"Shire": "Kontu"})
    assert g.confirm("Shire")
    assert g.terms["Shire"]["origin"] == "user"
    assert not g.confirm("Missing")


def test_delete(tmp_path):
    g = SeriesGlossary(name="LotR")
    g.set_user("Ring", "Sormus")
    assert g.delete("Ring")
    assert not g.delete("Ring")
    assert g.entries() == {}


def test_series_files_are_isolated(tmp_path):
    a = SeriesGlossary(name="LotR")
    a.set_user("Hobbit", "hobitti")
    a.save(series_path("LotR", tmp_path))
    b = SeriesGlossary(name="Dark")
    b.merge_auto({"Hobbit": "erihobitti"})
    b.save(series_path("Dark", tmp_path))
    assert SeriesGlossary.load(series_path("LotR", tmp_path)).entries()["Hobbit"] == "hobitti"
    assert SeriesGlossary.load(series_path("Dark", tmp_path)).entries()["Hobbit"] == "erihobitti"
    assert list_series(tmp_path) == ["Dark", "LotR"]


def test_legacy_flat_json_loads_as_user_terms(tmp_path):
    path = tmp_path / "Old.json"
    path.write_text(json.dumps({"Hobbit": "hobitti"}), encoding="utf-8")
    g = SeriesGlossary.load(path)
    assert g.terms["Hobbit"]["origin"] == "user"


def test_series_name_sanitized(tmp_path):
    with pytest.raises(ValueError):
        series_path("../escape", tmp_path)
    with pytest.raises(ValueError):
        series_path("has space", tmp_path)
    with pytest.raises(ValueError):
        series_path("", tmp_path)


def test_revision_changes_on_edit():
    g = SeriesGlossary(name="LotR")
    r1 = g.revision()
    g.merge_auto({"Ring": "Sormus"})
    assert g.revision() != r1


def _engine_with_series(tmp_path, series_terms=None, **cfg_kw):
    c = cfg(series="LotR", series_dir=str(tmp_path), **cfg_kw)
    if series_terms is not None:
        g = SeriesGlossary(name="LotR")
        for s, t in series_terms.items():
            g.set_user(s, t)
        g.save(series_path("LotR", tmp_path))
    return c


def test_engine_injects_series_terms(tmp_path):
    c = _engine_with_series(tmp_path, {"Palantir": "palantiri"})
    provider = MockProvider()
    report = translate_document_sync(make_doc(), c, provider)
    assert report.ok
    first_user_msg = provider.calls[0]["messages"][-1]["content"]
    assert "palantiri" in first_user_msg


def test_engine_saves_auto_candidates_post_run(tmp_path):
    c = _engine_with_series(tmp_path, {})
    provider = MockProvider(new_terms={"hyperdrive": "hyperajuri"})
    report = translate_document_sync(make_doc(), c, provider)
    assert report.ok
    g = SeriesGlossary.load(series_path("LotR", tmp_path))
    assert g.terms["hyperdrive"]["target"] == "hyperajuri"
    assert g.terms["hyperdrive"]["origin"] == "auto"  # candidate, not promoted


def test_static_glossary_beats_series(tmp_path):
    c = _engine_with_series(tmp_path, {"Hobbit": "hobitti"})
    static = tmp_path / "static.json"
    static.write_text(json.dumps({"Hobbit": "haltija"}), encoding="utf-8")
    c.translator.glossary_path = str(static)
    engine = TranslationEngine(c, MockProvider())
    assert engine._glossary.entries["Hobbit"] == "haltija"


def test_fingerprint_invalidated_by_series_edit(tmp_path):
    c = _engine_with_series(tmp_path, {"Ring": "Sormus"})
    doc = make_doc()
    fp1 = _fingerprint(TranslationEngine(c, MockProvider()), doc)
    # user corrects a term -> resume state must not be reused
    g = SeriesGlossary.load(series_path("LotR", tmp_path))
    g.set_user("Ring", "Rengas")
    g.save(series_path("LotR", tmp_path))
    fp2 = _fingerprint(TranslationEngine(c, MockProvider()), doc)
    assert fp1 != fp2


def test_candidates_labeled_as_suggestions_not_canonical(tmp_path):
    c = cfg(series="LotR", series_dir=str(tmp_path))
    g = SeriesGlossary(name="LotR")
    g.set_user("Hobbit", "hobitti")
    g.merge_auto({"Shire": "Kontu"})
    g.save(series_path("LotR", tmp_path))
    provider = MockProvider()
    translate_document_sync(make_doc(), c, provider)
    msg = provider.calls[0]["messages"][-1]["content"]
    glossary_line = next(line for line in msg.splitlines() if line.startswith("GLOSSARY:"))
    assert "hobitti" in glossary_line
    assert "Kontu" not in glossary_line, "candidates must not pose as canonical"
    assert "SUGGESTED TERMS" in msg
    suggested = msg.split("SUGGESTED TERMS", 1)[1]
    assert "Kontu" in suggested and "NOT user-confirmed" in msg
