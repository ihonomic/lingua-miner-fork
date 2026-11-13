import json
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main


def client(tmp_path):
    main.CON = main.db.connect(tmp_path / "t.db")
    return TestClient(main.app)


def _seg_session(text="Ets molt intel·ligent, tu."):
    return main.db.create_session(
        main.CON, title="V", source_type="local", media_path="/x.mp4",
        srt_source="whisper", model_size="small", duration_secs=10,
        transcript_json=json.dumps([{"start": 1.0, "end": 2.0, "text": text,
                                     "words": [], "logprob": 0.0,
                                     "tokens": []}]))


def test_senses_prefer_lemma_over_surface():
    d = main.dictionary.Dictionary(
        {"ets": [("ETS", "n")], "ser": [("ser", "vbser"), ("estar", "vblex")]})
    assert main._senses("ets", "ser", d)[0][0] == "ser"
    # multi-palabra: superficie primero
    d2 = main.dictionary.Dictionary({"poc a poc": [("despacio", "adv")]})
    assert main._senses("poc a poc", "poc a poc", d2) == [("despacio", "adv")]


def test_active_sense_matches_sentence_translation():
    senses = [("estar", "vblex"), ("perro", "n")]
    assert main._active_sense(senses, "El perro corre por ahí.", "") == 1
    assert main._active_sense(senses, "nada que ver", "perro") == 1
    assert main._active_sense(senses, "nada", "nada") == 0
    assert main._active_sense([], "x", "x") == -1


@patch("app.main.ipa.ipa", return_value="/əts/")
@patch("app.main.translate.sentence", side_effect=lambda t: "ES:" + t)
@patch("app.main.translate.translate",
       side_effect=lambda t: "eres" if t == "ets" else "T:" + t)
def test_lookup_word_es_uses_lowercase_form(_tr, _sen, _ipa, tmp_path,
                                            monkeypatch):
    from app import forms
    c = client(tmp_path)
    monkeypatch.setattr(forms, "known_exact", lambda w: False)
    monkeypatch.setattr(forms, "knows_lower", lambda w: w.lower() == "ets")
    r = c.post("/api/lookup",
               json={"selection": "Ets",
                     "sentence": "Ets molt intel·ligent, tu."}).json()
    assert r["word_es"] == "eres"
    assert r["ipa"] == "/əts/"
    assert "active" in r


@patch("app.main._flush", return_value=1)
@patch("app.main.media.animated_clip")
@patch("app.main.media.snapshot")
@patch("app.main.media.cut_audio")
@patch("app.main.translate.sentence", side_effect=lambda t: "ES:" + t)
@patch("app.main.translate.translate", side_effect=lambda t: "ES:" + t)
def test_mine_creates_and_flushes(_tr, _sen, _cut, _snap, _clip, _fl, tmp_path):
    c = client(tmp_path)
    sid = _seg_session("El gos corre")
    r = c.post("/api/cards/mine",
               json={"session_id": sid, "segment_index": 0,
                     "selection": "gos"}).json()
    assert r["sent_now"] is True
    assert r["paraula"] == "gos"
    assert r["word_status"] == "learning"
    rows = [dict(x) for x in main.CON.execute("SELECT * FROM cards")]
    assert any(cc["paraula"] == "gos" for cc in rows)


@patch("app.main._flush", return_value=0)
@patch("app.main.media.animated_clip")
@patch("app.main.media.snapshot")
@patch("app.main.media.cut_audio")
@patch("app.main.translate.sentence", side_effect=lambda t: "ES:" + t)
@patch("app.main.translate.translate", side_effect=lambda t: "ES:" + t)
def test_mine_respects_chosen_sense(_tr, _sen, _cut, _snap, _clip, _fl, tmp_path):
    c = client(tmp_path)
    sid = _seg_session("El gos corre")
    r = c.post("/api/cards/mine",
               json={"session_id": sid, "segment_index": 0,
                     "selection": "gos", "paraula_es": "perro"}).json()
    assert r["sent_now"] is False
    row = main.CON.execute("SELECT paraula_es FROM cards").fetchone()
    assert row["paraula_es"] == "perro"


@patch("app.main._definition")
@patch("app.main._flush", return_value=1)
@patch("app.main.media.animated_clip")
@patch("app.main.media.snapshot")
@patch("app.main.media.cut_audio")
@patch("app.main.translate.sentence", side_effect=lambda t: "ES:" + t)
@patch("app.main.translate.translate", side_effect=lambda t: "ES:" + t)
def test_mine_whole_sentence(_tr, _sen, _cut, _snap, _clip, _fl, _def, tmp_path):
    # ⇧ derecha: la frase entera, sin palabra elegida
    c = client(tmp_path)
    sid = _seg_session("El gos corre")
    r = c.post("/api/cards/mine",
               json={"session_id": sid, "segment_index": 0, "whole": True}).json()
    assert r["sent_now"] is True and r["paraula"] == "El gos corre"
    row = dict(main.CON.execute("SELECT * FROM cards").fetchone())
    assert row["paraula_es"] == row["frase_es"] == "ES:El gos corre"
    assert row["lema"] == "" and row["freq_rank"] == "" and row["definicio"] == ""
    # ni entra en la lista de palabras ni se busca en el diccionario
    assert r["word_status"] is None
    _def.assert_not_called()
    _tr.assert_not_called()
    # en Anki va una vez, no dos
    f = main.anki.build_note(row, "D")["fields"]
    assert f["Paraula"] == "El gos corre" and f["ParaulaES"] == "ES:El gos corre"
    assert f["Frase"] == "" and f["FraseES"] == ""


@patch("app.main.anki.is_up", return_value=False)
def test_stats_local_only_when_anki_down(_up, tmp_path):
    c = client(tmp_path)
    sid = _seg_session("El gos corre")
    main.db.create_card(
        main.CON, session_id=sid, segment_index=0, paraula="gos",
        lema="gos", pos="NOUN", paraula_es="perro", frase="El gos corre",
        frase_es="El perro corre", freq_rank="common", audio_file="",
        image_file="", font="V")
    main.db.set_word_status(main.CON, "gos", "learning")
    main.db.set_word_status(main.CON, "casa", "known")
    r = c.get("/api/stats").json()
    assert r["total_cards"] == 1
    assert r["status_counts"]["learning"] == 1
    assert r["status_counts"]["known"] == 1
    assert len(r["by_month"]) == 1
    assert r["anki"] is None


def test_sentence_key_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(main.config, "SETTINGS_PATH", tmp_path / "s.json")
    c = client(tmp_path)
    assert c.get("/api/settings").json()["sentence_key"] == "ShiftRight"
    assert c.post("/api/settings", json={"sentence_key": "AltRight"}).status_code == 200
    assert c.get("/api/settings").json()["sentence_key"] == "AltRight"
    assert c.post("/api/settings", json={"sentence_key": "F5"}).status_code == 200
    # «-» y «#» de un teclado alemán
    for ok in ("Slash", "Backslash"):
        assert c.post("/api/settings", json={"sentence_key": ok}).status_code == 200
    # ni teclas fijas, ni una letra del mapa, ni basura
    for bad in ("Space", "ArrowLeft", "KeyQ", "", "<script>"):
        assert c.post("/api/settings", json={"sentence_key": bad}).status_code == 400
    # tampoco al revés: una letra libre como tecla de frase bloquea esa letra
    assert c.post("/api/settings", json={"sentence_key": "KeyX"}).status_code == 200
    assert c.post("/api/settings", json={"keymap": {"mine": "x"}}).status_code == 400


@patch("app.main.anki.ensure_note_type")
@patch("app.main.anki.is_up", return_value=True)
def test_sync_statuses_settles_with_sentence_cards(_up, _ens, tmp_path, monkeypatch):
    # Antes: cada sync contaba la tarjeta de frase (sin lema) como cambio
    c = client(tmp_path)
    sid = _seg_session("El gos corre")
    for i, (paraula, lema) in enumerate([("El gos corre", ""), ("Gos", "Gos")]):
        cid = main.db.create_card(
            main.CON, session_id=sid, segment_index=0, paraula=paraula,
            lema=lema, pos="", paraula_es="", frase="El gos corre",
            frase_es="", freq_rank="", audio_file="", image_file="", font="",
            language=main._lang())
        main.db.mark_card_sent(main.CON, cid, 100 + i)
    monkeypatch.setattr(main.anki, "note_intervals", lambda ids: {i: 3 for i in ids})
    assert c.post("/api/anki/sync-statuses").json()["synced"] == 1
    assert c.post("/api/anki/sync-statuses").json()["synced"] == 0


@patch("app.main._definition")
@patch("app.main._flush", return_value=1)
@patch("app.main.media.animated_clip")
@patch("app.main.media.snapshot")
@patch("app.main.media.cut_audio")
@patch("app.main.translate.sentence", side_effect=lambda t: "ES:" + t)
@patch("app.main.translate.translate", side_effect=lambda t: "ES:" + t)
def test_preview_whole_sentence_then_send(_tr, _sen, _cut, _snap, _clip, _fl, _def, tmp_path):
    # ↑: el editor se abre con la frase entera y su traducción, sin palabra
    c = client(tmp_path)
    sid = _seg_session("El gos corre")
    p = c.post("/api/cards/preview",
               json={"session_id": sid, "segment_index": 0, "whole": True}).json()
    assert p["paraula"] == p["frase"] == "El gos corre"
    assert p["paraula_es"] == p["frase_es"] == "ES:El gos corre"
    assert p["lema"] == "" and p["senses"] == [] and p["freq_rank"] == ""
    _tr.assert_not_called()
    r = c.post("/api/cards", json={k: p[k] for k in (
        "paraula", "lema", "pos", "paraula_es", "frase", "frase_es",
        "freq_rank", "audio_file", "image_file", "font")} |
        {"session_id": sid, "segment_index": 0}).json()
    assert r["word_status"] is None
    _def.assert_not_called()
