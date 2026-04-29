from scripts.eval_teacher_forcing import _format_category_stats


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
