from collections import Counter

from scripts.eval_teacher_forcing import _format_category_stats, _format_prediction_counts, _token_family


def test_format_category_stats_reports_top1_and_top5_rates():
    stats = _format_category_stats(
        {
            "MEL_INT12": {"count": 4, "top1": 3, "top5": 4},
            "HARM_CLASS": {"count": 0, "top1": 0, "top5": 0},
        }
    )

    assert stats["MEL_INT12"] == {
        "count": 4,
        "top1_accuracy": 0.75,
        "top5_accuracy": 1.0,
    }
    assert stats["HARM_CLASS"] == {
        "count": 0,
        "top1_accuracy": None,
        "top5_accuracy": None,
    }


def test_token_family_splits_chorale_v2_voice_tokens():
    assert _token_family("BASS_48") == "BASS"
    assert _token_family("TENOR_60") == "TENOR"
    assert _token_family("ALTO_64") == "ALTO"
    assert _token_family("SOP_72") == "SOP"
    assert _token_family("POS_24") == "POS"
    assert _token_family("DUR_12") == "DUR"
    assert _token_family("STYLE_CHORALE") == "META"


def test_format_prediction_counts_limits_common_tokens():
    counts = _format_prediction_counts(
        {"SOP": Counter({"SOP_72": 5, "SOP_71": 3, "SOP_69": 1})},
        limit=2,
    )

    assert counts == {
        "SOP": [
            {"token": "SOP_72", "count": 5},
            {"token": "SOP_71", "count": 3},
        ]
    }
