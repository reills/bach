import argparse
import copy
import re
from pathlib import Path

from music21 import converter


def _write_score(score, out_path: Path) -> None:
    written = score.write("musicxml", fp=str(out_path))
    written_path = Path(written)
    if written_path != out_path and written_path.exists():
        written_path.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_dir = (
        Path(args.root)
        / "data"
        / "tobis_xml"
        / "instrumental-works"
        / "Art of fugue"
    )
    if not src_dir.exists():
        raise SystemExit(f"Missing: {src_dir}")

    pattern = re.compile(r"^artfugue-(\d+)([a-z]?)$")
    for krn_path in sorted(src_dir.glob("artfugue-*.krn")):
        match = pattern.match(krn_path.stem)
        if not match:
            print(f"skip (name): {krn_path.name}")
            continue

        num = int(match.group(1))
        suffix = match.group(2)
        bwv_name = f"BWV_1080_{num:02d}{suffix}"
        out_dir = src_dir / bwv_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{bwv_name}.xml"

        if out_path.exists() and not args.overwrite:
            print(f"exists: {out_path}")
            continue

        try:
            score = converter.parse(krn_path)

            # Try writing the score with multiple fallback strategies
            for attempt in range(3):
                try:
                    _write_score(score, out_path)
                    print(f"wrote: {out_path}")
                    break
                except Exception as exc:
                    if "already found in this Stream" in str(exc):
                        if attempt == 0:
                            # First attempt: try coreCopy
                            if hasattr(score, "coreCopy"):
                                score = score.coreCopy()
                            else:
                                score = copy.deepcopy(score)
                        elif attempt == 1:
                            # Second attempt: deep copy and recreate
                            score = copy.deepcopy(score)
                            # Clear any cached IDs
                            for site in score.recurse():
                                site.id = None
                        else:
                            # Final attempt failed
                            raise
                    else:
                        # Different error, raise immediately
                        raise
        except Exception as exc:
            print(f"error: {krn_path.name}: {exc}")


if __name__ == "__main__":
    main()
