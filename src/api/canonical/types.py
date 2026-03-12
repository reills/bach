from dataclasses import dataclass, field


def _validate_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")


def _validate_tick_map(name: str, tick_map: dict[int, object]) -> None:
    for tick in tick_map:
        _validate_int(f"{name} ticks", tick)
        if tick < 0:
            raise ValueError(f"{name} ticks must be non-negative")


@dataclass(frozen=True)
class ScoreHeader:
    tpq: int
    key_sig_map: dict[int, str] = field(default_factory=dict)
    time_sig_map: dict[int, str] = field(default_factory=dict)
    tempo_map: dict[int, int] = field(default_factory=dict)
    pickup_ticks: int = 0

    def __post_init__(self) -> None:
        _validate_int("tpq", self.tpq)
        if self.tpq <= 0:
            raise ValueError("tpq must be positive")
        _validate_int("pickup_ticks", self.pickup_ticks)
        if self.pickup_ticks < 0:
            raise ValueError("pickup_ticks must be non-negative")
        if 0 not in self.time_sig_map:
            raise ValueError("time_sig_map must define a time signature at tick 0")
        _validate_tick_map("key_sig_map", self.key_sig_map)
        _validate_tick_map("time_sig_map", self.time_sig_map)
        _validate_tick_map("tempo_map", self.tempo_map)

        for tick, time_sig in self.time_sig_map.items():
            if not time_sig:
                raise ValueError(f"time_sig_map[{tick}] must be non-empty")
        for tick, key_sig in self.key_sig_map.items():
            if not key_sig:
                raise ValueError(f"key_sig_map[{tick}] must be non-empty")
        for tick, tempo in self.tempo_map.items():
            _validate_int(f"tempo_map[{tick}]", tempo)
            if tempo <= 0:
                raise ValueError(f"tempo_map[{tick}] must be positive")


@dataclass(frozen=True)
class PartInfo:
    id: str
    instrument: str
    tuning: list[int] = field(default_factory=list)
    capo: int = 0
    midi_program: int | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("part id must be non-empty")
        if not self.instrument:
            raise ValueError("instrument must be non-empty")
        _validate_int("capo", self.capo)
        if self.capo < 0:
            raise ValueError("capo must be non-negative")
        if self.midi_program is not None:
            _validate_int("midi_program", self.midi_program)
            if not 0 <= self.midi_program <= 127:
                raise ValueError("midi_program must be between 0 and 127")
        for pitch in self.tuning:
            _validate_int("tuning pitches", pitch)
            if not 0 <= pitch <= 127:
                raise ValueError("tuning pitches must be valid MIDI values")


@dataclass(frozen=True)
class Measure:
    id: str
    index: int
    start_tick: int
    length_ticks: int

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("measure id must be non-empty")
        _validate_int("measure index", self.index)
        if self.index < 0:
            raise ValueError("measure index must be non-negative")
        _validate_int("measure start_tick", self.start_tick)
        if self.start_tick < 0:
            raise ValueError("measure start_tick must be non-negative")
        _validate_int("measure length_ticks", self.length_ticks)
        if self.length_ticks <= 0:
            raise ValueError("measure length_ticks must be positive")

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.length_ticks


@dataclass(frozen=True)
class GuitarFingering:
    string_index: int
    fret: int

    def __post_init__(self) -> None:
        _validate_int("string_index", self.string_index)
        if self.string_index < 0:
            raise ValueError("string_index must be non-negative")
        _validate_int("fret", self.fret)
        if self.fret < 0:
            raise ValueError("fret must be non-negative")


@dataclass(frozen=True)
class Event:
    id: str
    start_tick: int
    dur_tick: int
    voice_id: int
    pitch_midi: int | None
    velocity: int | None = None
    fingering: GuitarFingering | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("event id must be non-empty")
        _validate_int("event start_tick", self.start_tick)
        if self.start_tick < 0:
            raise ValueError("event start_tick must be non-negative")
        _validate_int("event dur_tick", self.dur_tick)
        if self.dur_tick <= 0:
            raise ValueError("event dur_tick must be positive")
        _validate_int("voice_id", self.voice_id)
        if self.voice_id < 0:
            raise ValueError("voice_id must be non-negative")
        if self.pitch_midi is not None:
            _validate_int("pitch_midi", self.pitch_midi)
            if not 0 <= self.pitch_midi <= 127:
                raise ValueError("pitch_midi must be a valid MIDI value or None")
        if self.velocity is not None:
            _validate_int("velocity", self.velocity)
            if not 1 <= self.velocity <= 127:
                raise ValueError("velocity must be between 1 and 127")
        if self.pitch_midi is None and self.fingering is not None:
            raise ValueError("rests cannot carry fingering data")

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.dur_tick


@dataclass(frozen=True)
class Part:
    info: PartInfo
    events: list[Event] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.info, PartInfo):
            raise ValueError("part info must be a PartInfo instance")
        for event in self.events:
            if not isinstance(event, Event):
                raise ValueError("part events must be Event instances")


@dataclass(frozen=True)
class CanonicalScore:
    header: ScoreHeader
    measures: list[Measure]
    parts: list[Part]

    def __post_init__(self) -> None:
        if not isinstance(self.header, ScoreHeader):
            raise ValueError("header must be a ScoreHeader instance")
        if not self.measures:
            raise ValueError("canonical score requires at least one measure")
        if not self.parts:
            raise ValueError("canonical score requires at least one part")

        measure_ids: set[str] = set()
        expected_start_tick = 0
        for expected_index, measure in enumerate(self.measures):
            if not isinstance(measure, Measure):
                raise ValueError("measures must contain Measure instances")
            if measure.id in measure_ids:
                raise ValueError(f"duplicate measure id: {measure.id}")
            if measure.index != expected_index:
                raise ValueError("measure indices must be contiguous and match list order")
            if measure.start_tick != expected_start_tick:
                raise ValueError("measures must be contiguous in tick order")
            measure_ids.add(measure.id)
            expected_start_tick = measure.end_tick

        total_ticks = expected_start_tick
        part_ids: set[str] = set()
        event_ids: set[str] = set()

        for part in self.parts:
            if not isinstance(part, Part):
                raise ValueError("parts must contain Part instances")
            part_id = part.info.id
            if part_id in part_ids:
                raise ValueError(f"duplicate part id: {part_id}")
            part_ids.add(part_id)

            previous_start_tick = -1
            voice_ids: set[int] = set()
            for event in part.events:
                if event.id in event_ids:
                    raise ValueError(f"duplicate event id: {event.id}")
                if event.start_tick < previous_start_tick:
                    raise ValueError("part events must be sorted by start_tick")
                if event.start_tick >= total_ticks:
                    raise ValueError("event start_tick must land inside the score")
                if event.end_tick > total_ticks:
                    raise ValueError("event dur_tick must stay inside the score")

                self.measure_for_tick(event.start_tick)
                event_ids.add(event.id)
                previous_start_tick = event.start_tick
                voice_ids.add(event.voice_id)

            if voice_ids != set(range(len(voice_ids))):
                raise ValueError("part voice_ids must be contiguous per part starting at 0")

    @property
    def total_ticks(self) -> int:
        return self.measures[-1].end_tick

    def measure_for_tick(self, tick: int) -> Measure:
        _validate_int("tick", tick)
        if tick < 0:
            raise ValueError("tick must be non-negative")
        for measure in self.measures:
            if measure.start_tick <= tick < measure.end_tick:
                return measure
        raise ValueError(f"tick {tick} is outside the score")
