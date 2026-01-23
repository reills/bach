import sys
from pathlib import Path
import pytest
from music21 import corpus

# Ensure src is in path if running from root
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.tokens.eventizer import eventize_musicxml
from src.tokens.roundtrip import tokens_to_midi
from src.tokens.validator import validate_harm_tokens

def _local_xml(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"Local file {path} not found")
    return str(path)

@pytest.fixture
def bwv66_xml(tmp_path):
    """Creates a temporary MusicXML file for BWV 66.6."""
    piece = corpus.parse("bach/bwv66.6")
    xml_path = tmp_path / "bwv66.6.xml"
    piece.write("musicxml", fp=str(xml_path))
    return str(xml_path)

@pytest.fixture
def lute_suite_xml():
    """Uses the locally provided BWV 995 file."""
    path = Path("data/tobis_xml/instrumental-works/Lute works/BWV_0995/BWV_0995.xml")
    return _local_xml(path)

@pytest.fixture
def cello_suite_xml():
    """Uses the locally provided BWV 1007 file."""
    path = Path("data/tobis_xml/instrumental-works/chamber-music/BWV 1007-1012 Suites for Solo Cello/BWV_1007/BWV_1007.musicxml")
    return _local_xml(path)

@pytest.fixture
def bwv552_xml():
    """Uses the locally provided BWV 552 file."""
    path = Path("data/tobis_xml/instrumental-works/organ-works/BWV 531-552 Preludes and Fugues/BWV_0552/BWV_0552.xml")
    return _local_xml(path)

@pytest.fixture
def bwv686_xml():
    """Uses the locally provided BWV 686 file."""
    path = Path("data/tobis_xml/instrumental-works/organ-works/BWV 669-689 Chorale Preludes  German Organ Mass/BWV_0686/BWV_0686.xml")
    return _local_xml(path)

@pytest.fixture
def bwv1079_xml():
    """Uses the locally provided BWV 1079 file."""
    path = Path("data/tobis_xml/instrumental-works/Musical offering/BWV_1079_01/BWV_1079_01.xml")
    return _local_xml(path)

@pytest.fixture
def bwv225_xml():
    """Uses the locally provided BWV 225 file."""
    path = Path("data/tobis_xml/vocal-works/Motets/BWV_0225/BWV_0225.xml")
    return _local_xml(path)

@pytest.fixture
def bwv226_xml():
    """Uses the locally provided BWV 226 file."""
    path = Path("data/tobis_xml/vocal-works/Motets/BWV_0226/BWV_0226.xml")
    return _local_xml(path)

@pytest.fixture
def bwv228_xml():
    """Uses the locally provided BWV 228 file."""
    path = Path("data/tobis_xml/vocal-works/Motets/BWV_0228/BWV_0228.xml")
    return _local_xml(path)

@pytest.fixture
def bwv232_xml():
    """Uses the locally provided BWV 232 file."""
    path = Path("data/tobis_xml/vocal-works/Masses/BWV_0232_01/BWV_0232_01.musicxml")
    return _local_xml(path)

@pytest.fixture
def bwv243_xml():
    """Uses the locally provided BWV 243 file."""
    path = Path("data/tobis_xml/vocal-works/Magnificat/BWV_0243_d_D/BWV_0243_d_D.musicxml")
    return _local_xml(path)

@pytest.fixture
def bwv244_xml():
    """Uses the locally provided BWV 244 file."""
    path = Path("data/tobis_xml/vocal-works/passions/st-matthew-passion/BWV_0244_01/BWV_0244_01.xml")
    return _local_xml(path)

@pytest.fixture
def bwv1052_xml():
    """Uses the locally provided BWV 1052 file."""
    path = Path("data/tobis_xml/instrumental-works/concertos/BWV 1052-1065 Harpsichord Concertos/BWV_1052_1/BWV_1052_1.xml")
    return _local_xml(path)

@pytest.fixture
def bwv830_xml():
    """Uses the locally provided BWV 830 file."""
    path = Path("data/tobis_xml/instrumental-works/keyboard-works/BWV 825-831 Partitas/BWV_0830/BWV_0830.xml")
    return _local_xml(path)

@pytest.fixture
def bwv1068_xml():
    """Uses the locally provided BWV 1068 file."""
    path = Path("data/tobis_xml/instrumental-works/Overtures sinfonias/BWV_1068_01/BWV_1068_01.xml")
    return _local_xml(path)

@pytest.fixture
def bwv1047_xml():
    """Uses the locally provided BWV 1047 file."""
    path = Path("data/tobis_xml/instrumental-works/concertos/BWV 1046-1051 Brandenburg Concertos/BWV_1047_1/BWV_1047_1.xml")
    return _local_xml(path)

@pytest.fixture
def bwv1080_xml():
    """Uses the locally provided BWV 1080 file."""
    path = Path("data/tobis_xml/instrumental-works/Art of fugue/BWV_1080_01/BWV_1080_01.xml")
    return _local_xml(path)

@pytest.fixture
def bwv_anh_078_xml():
    """Uses the locally provided BWV Anh. 78 file."""
    path = Path("data/tobis_xml/supplement/Supplement II  Doubtful works/BWV_Anh_078/BWV_Anh_078.xml")
    return _local_xml(path)

@pytest.fixture
def bwv1080_05_xml():
    """Uses the locally provided BWV 1080/5 file."""
    path = Path("data/tobis_xml/instrumental-works/Art of fugue/BWV_1080_05/BWV_1080_05.xml")
    return _local_xml(path)

@pytest.fixture
def bwv0061_03_xml():
    """Uses the locally provided BWV 61/3 file."""
    path = Path("data/tobis_xml/vocal-works/Cantatas/BWV 061-080/BWV_0061_03/BWV_0061_03.xml")
    return _local_xml(path)

@pytest.fixture
def bwv0092_03_xml():
    """Uses the locally provided BWV 92/3 file."""
    path = Path("data/tobis_xml/vocal-works/Cantatas/BWV 081-100/BWV_0092_03/BWV_0092_03.xml")
    return _local_xml(path)

@pytest.fixture
def bwv1063_2_xml():
    """Uses the locally provided BWV 1063/2 file."""
    path = Path("data/tobis_xml/instrumental-works/concertos/BWV 1052-1065 Harpsichord Concertos/BWV_1063_2/BWV_1063_2.xml")
    return _local_xml(path)

@pytest.fixture
def bwv0656a_xml():
    """Uses the locally provided BWV 656a file."""
    path = Path("data/tobis_xml/instrumental-works/organ-works/BWV 651-668 Leipzig Chorale Preludes/BWV_0656a/BWV_0656a.xml")
    return _local_xml(path)

@pytest.fixture
def bwv0016_03_xml():
    """Uses the locally provided BWV 16/3 file."""
    path = Path("data/tobis_xml/vocal-works/Cantatas/BWV 001-020/BWV_0016_03/BWV_0016_03.xml")
    return _local_xml(path)

@pytest.fixture
def bwv0197a_xml():
    """Uses the locally provided BWV 197a file."""
    path = Path("data/tobis_xml/vocal-works/Cantatas/BWV 181-200/BWV_0197a/BWV_0197a.xml")
    return _local_xml(path)

@pytest.mark.parametrize(
    "fixture_name",
    [
        "bwv66_xml",
        "lute_suite_xml",
        "cello_suite_xml",
        "bwv552_xml",
        "bwv686_xml",
        "bwv1079_xml",
        "bwv225_xml",
        "bwv226_xml",
        "bwv228_xml",
        "bwv232_xml",
        "bwv243_xml",
        "bwv244_xml",
        "bwv1052_xml",
        "bwv830_xml",
        "bwv1068_xml",
        "bwv1047_xml",
        "bwv1080_xml",
        "bwv_anh_078_xml",
        "bwv1063_2_xml",
        "bwv0656a_xml",
        "bwv0016_03_xml",
        "bwv0197a_xml",
        "bwv1080_05_xml",
        "bwv0061_03_xml",
        "bwv0092_03_xml",
    ],
)
def test_piece_tokenization_integrity(fixture_name, request):
    """
    Integration test for the tokenizer pipeline across different styles.
    """
    xml_path = request.getfixturevalue(fixture_name)
    tokens, meta = eventize_musicxml(xml_path)
    
    # 1. Basic Structure Checks
    assert len(tokens) > 50, f"Token stream for {fixture_name} is suspiciously short"
    assert "BAR" in tokens
    
    # 2. Validate HARM Invariants
    errors = validate_harm_tokens(tokens)
    
    error_msg = "\n".join(errors[:10])
    if len(errors) > 10:
        error_msg += f"\n... and {len(errors) - 10} more errors."
        
    assert not errors, f"HARM token validation failed for {fixture_name} with {len(errors)} errors:\n{error_msg}"

def test_roundtrip_midi_generation(bwv66_xml, tmp_path):
    """
    Smoke test to ensure tokens can be converted back to MIDI without crashing.
    """
    tokens, _ = eventize_musicxml(bwv66_xml)
    midi_out = tmp_path / "roundtrip.mid"
    
    # Should run without error
    tokens_to_midi(tokens, str(midi_out))
    
    assert midi_out.exists()
    assert midi_out.stat().st_size > 0
