type SliderInputProps = {
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
}

export function SliderInput({ value, onChange, min, max, step }: SliderInputProps) {
  return (
    <div className="flex items-center gap-3">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 accent-neutral-900"
      />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-24 shrink-0 rounded border border-neutral-300 px-2 py-1 text-sm focus:border-neutral-500 focus:outline-none"
      />
    </div>
  )
}
