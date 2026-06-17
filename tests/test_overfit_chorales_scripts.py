import pandas as pd

from scripts.eval_overfit_continuation import _pass_flags, _tokens_to_bars
from scripts.prepare_overfit_chorales import CHORALE_SOURCE_FRAGMENT, select_clean_pieces


def test_select_clean_pieces_filters_to_four_part_chorales():
    rows = []
    for piece_idx in range(3):
        for bar_idx in range(2):
            rows.append(
                {
                    "piece_id": f"chorale_{piece_idx}",
                    "source_path": f"{CHORALE_SOURCE_FRAGMENT}/BWV_{piece_idx}/score.xml",
                    "bar_index": bar_idx,
                    "tokens": "BAR POS_0 VOICE_0 DUR_24 VOICE_1 DUR_24 VOICE_2 DUR_24 VOICE_3 DUR_24",
                }
            )
    rows.append(
        {
            "piece_id": "tab_piece",
            "source_path": f"{CHORALE_SOURCE_FRAGMENT}/tab/score.xml",
            "bar_index": 0,
            "tokens": "BAR POS_0 VOICE_0 DUR_24 VOICE_1 DUR_24 VOICE_2 DUR_24 VOICE_3 DUR_24 STR_1",
        }
    )
    rows.append(
        {
            "piece_id": "keyboard_piece",
            "source_path": "instrumental-works/keyboard-works/BWV_0846/BWV_0846.xml",
            "bar_index": 0,
            "tokens": "BAR POS_0 VOICE_0 DUR_24 VOICE_1 DUR_24 VOICE_2 DUR_24 VOICE_3 DUR_24",
        }
    )

    selected, pieces = select_clean_pieces(
        pd.DataFrame(rows),
        limit=2,
        min_bars=2,
        min_pct_4plus=1.0,
    )

    assert selected["piece_id"].nunique() == 2
    assert len(pieces) == 2
    assert set(selected["source_path"].str.contains(CHORALE_SOURCE_FRAGMENT)) == {True}
    assert "tab_piece" not in set(selected["piece_id"])
    assert "keyboard_piece" not in set(selected["piece_id"])


def test_eval_helpers_report_requested_bars_and_quality_flags():
    bars = _tokens_to_bars(["KEY_C", "BAR", "POS_0", "VOICE_0", "DUR_24", "BAR", "POS_0"])
    assert bars == [["BAR", "POS_0", "VOICE_0", "DUR_24"], ["BAR", "POS_0"]]

    flags = _pass_flags(
        {
            "avg_voices_per_bar": 4.0,
            "pct_bars_3plus_voices": 100.0,
            "duplicate_bar_rate": 0.0,
            "cadence_proxy_rate": 0.75,
            "token_grammar_violations": 0,
        },
        0.5,
        generated_continuation_bar_count=6,
        requested_continuation_bars=6,
    )

    assert flags == {
        "generates_requested_bars": True,
        "recognizable_token_overlap": True,
        "keeps_4_voice_texture": True,
        "cadence_proxy": True,
        "avoids_obvious_repetition": True,
    }
