interface ZoomControlProps {
  value: number;
  onChange: (value: number) => void;
}

export function ZoomControl({ value, onChange }: ZoomControlProps) {
  return (
    <div className="zoom-control">
      <div className="zoom-control-heading">
        <label htmlFor="scene-zoom">Przybliżenie</label>
        <output htmlFor="scene-zoom">{Math.round(value)}%</output>
      </div>
      <input
        id="scene-zoom"
        type="range"
        min="0"
        max="100"
        step="1"
        value={value}
        aria-label="Przybliżenie widoku"
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      />
    </div>
  );
}