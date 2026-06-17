interface FingeringOption {
  stringIndex: number;
  fret: number;
  selected: boolean;
}

interface FingeringPickerProps {
  options: FingeringOption[];
  onSelect: (stringIndex: number, fret: number) => void;
  onClose: () => void;
  busy: boolean;
}

const FingeringPicker = ({ options, onSelect, onClose, busy }: FingeringPickerProps) => (
  <div className="fingering-picker">
    <div className="fingering-picker__header">
      <span className="fingering-picker__title">Alternate Positions</span>
      <button
        className="btn btn--icon-only"
        onClick={onClose}
        aria-label="Close fingering picker"
      >
        ✕
      </button>
    </div>
    <div className="fingering-picker__options">
      {options.map((opt) => (
        <button
          key={`${opt.stringIndex}-${opt.fret}`}
          className={`fingering-option${opt.selected ? ' fingering-option--current' : ''}`}
          onClick={() => onSelect(opt.stringIndex, opt.fret)}
          disabled={busy}
          title={`String ${opt.stringIndex + 1}, fret ${opt.fret}`}
        >
          <span className="fingering-option__pos">S{opt.stringIndex + 1} fr.{opt.fret}</span>
          {opt.selected ? <span className="fingering-option__check"> ✓</span> : null}
        </button>
      ))}
    </div>
  </div>
);

export default FingeringPicker;
