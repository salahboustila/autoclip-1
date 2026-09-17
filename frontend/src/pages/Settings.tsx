import { useEffect, useRef, useState } from 'react'

import {
  api,
  type ProviderStatus,
  type Settings as SettingsData,
  type SystemStatus,
  type WatermarkPosition,
} from '../api'
import { ErrorNote } from '../components/ErrorNote'

const WATERMARK_POSITIONS: { value: WatermarkPosition; label: string }[] = [
  { value: 'top-left', label: 'Top left' },
  { value: 'top-right', label: 'Top right' },
  { value: 'center', label: 'Center' },
  { value: 'bottom-left', label: 'Bottom left' },
  { value: 'bottom-right', label: 'Bottom right' },
]

const SECRET_LABELS: Record<string, string> = {
  anthropic: 'Anthropic API key',
  openai: 'OpenAI-compatible API key',
  gemini: 'Google Gemini API key',
  huggingface_token: 'HuggingFace token',
}

export function Settings() {
  const [settings, setSettings] = useState<SettingsData | null>(null)
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [system, setSystem] = useState<SystemStatus | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [saved, setSaved] = useState(false)
  const [watermarkPreviewUrl, setWatermarkPreviewUrl] = useState<string | null>(null)
  const [watermarkBusy, setWatermarkBusy] = useState(false)
  const watermarkFileInput = useRef<HTMLInputElement>(null)

  const reload = () => {
    void api
      .getSettings()
      .then((loaded) => {
        setSettings(loaded)
        // The image is only fetched once here (and again after an upload) —
        // not derived from `loaded` on every render — so a position/opacity
        // tweak doesn't refetch the same picture on every keystroke.
        setWatermarkPreviewUrl(loaded.export.watermark.filename ? api.watermarkFileUrl() : null)
      })
      .catch((e) => setError(e as Error))
    void api.providerStatus().then(setProviders).catch(() => undefined)
    void api.system().then(setSystem).catch(() => undefined)
  }

  useEffect(reload, [])

  const patch = async (update: Partial<SettingsData>) => {
    setError(null)
    try {
      setSettings(await api.putSettings(update))
      setSaved(true)
      setTimeout(() => setSaved(false), 1600)
    } catch (err) {
      setError(err as Error)
    }
  }

  /** Commits a change to just the watermark's own fields, spreading the rest
   * of `export` and of `watermark` so the shallow section-replace that
   * `PUT /api/settings` does for `export` can't drop an unrelated field. */
  const patchWatermark = (update: Partial<SettingsData['export']['watermark']>) => {
    if (!settings) return
    return patch({
      export: { ...settings.export, watermark: { ...settings.export.watermark, ...update } },
    })
  }

  const uploadWatermark = async (file: File) => {
    setWatermarkBusy(true)
    setError(null)
    try {
      const updated = await api.uploadWatermark(file)
      setSettings(updated)
      setWatermarkPreviewUrl(api.watermarkFileUrl())
    } catch (err) {
      setError(err as Error)
    } finally {
      setWatermarkBusy(false)
    }
  }

  const removeWatermark = async () => {
    setWatermarkBusy(true)
    setError(null)
    try {
      setSettings(await api.deleteWatermark())
      setWatermarkPreviewUrl(null)
    } catch (err) {
      setError(err as Error)
    } finally {
      setWatermarkBusy(false)
    }
  }

  if (!settings) return <p className="pt-24 text-sm text-ink-500">Loading…</p>

  return (
    <div className="max-w-4xl pt-14">
      <div className="rise flex items-baseline justify-between border-b border-ink-800 pb-5">
        <h1 className="font-display text-[clamp(2rem,4vw,3rem)] leading-none text-ink-100">
          Settings
        </h1>
        <span
          className="text-xs text-signal-good transition-opacity duration-300"
          style={{ opacity: saved ? 1 : 0 }}
        >
          saved
        </span>
      </div>

      {error && (
        <div className="mt-6">
          <ErrorNote error={error} onDismiss={() => setError(null)} />
        </div>
      )}

      {settings.insecure_secret_storage && (
        <p className="mt-6 border-l-2 border-sodium-600 pl-4 text-sm leading-relaxed text-ink-300">
          No OS keyring is available on this machine, so API keys are stored in plain text
          in <code className="text-ink-200">config.json</code>. On headless Linux, installing
          a Secret Service provider or <code className="text-ink-200">keyrings.alt</code>{' '}
          fixes this.
        </p>
      )}

      <Section title="AI provider" note="Which model picks the clips.">
        <div className="space-y-1">
          {providers.map((provider) => (
            <button
              key={provider.name}
              onClick={() => patch({ active_provider: provider.name })}
              className={[
                'block w-full border-l-2 py-3 pl-3 text-left transition-colors duration-200',
                provider.name === settings.active_provider
                  ? 'border-sodium-500 bg-ink-850/60'
                  : 'border-transparent hover:border-ink-700 hover:bg-ink-850/30',
              ].join(' ')}
            >
              <div className="flex items-baseline justify-between gap-4">
                <span className="text-sm text-ink-100">{provider.name}</span>
                <span
                  className={`text-xs ${provider.available ? 'text-signal-good' : 'text-ink-500'}`}
                >
                  {provider.available ? 'reachable' : provider.detail || 'unavailable'}
                </span>
              </div>
              {provider.models.length > 0 && (
                <span className="numeric mt-1 block truncate text-xs text-ink-600">
                  {provider.models.slice(0, 6).join(' · ')}
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="mt-6 grid gap-5 sm:grid-cols-2">
          <Field
            label="Model"
            value={settings.providers[settings.active_provider]?.model ?? ''}
            placeholder="model name"
            onCommit={(value) =>
              patch({
                providers: {
                  [settings.active_provider]: {
                    ...settings.providers[settings.active_provider],
                    model: value,
                  },
                },
              })
            }
          />
          <Field
            label="Base URL"
            hint="Point at OpenRouter, Groq, DeepSeek, or a local server."
            value={settings.providers[settings.active_provider]?.base_url ?? ''}
            placeholder="https://…"
            onCommit={(value) =>
              patch({
                providers: {
                  [settings.active_provider]: {
                    ...settings.providers[settings.active_provider],
                    base_url: value || null,
                  },
                },
              })
            }
          />
        </div>
      </Section>

      <Section title="Keys" note="Stored in your OS keyring. Never sent anywhere but the provider.">
        <div className="space-y-4">
          {Object.entries(SECRET_LABELS).map(([key, label]) => (
            <SecretField
              key={key}
              secretKey={key}
              label={label}
              present={settings.keys_present[key] ?? false}
              onChanged={reload}
              onError={setError}
            />
          ))}
        </div>
      </Section>

      <Section title="Transcription">
        <div className="grid gap-5 sm:grid-cols-2">
          <Select
            label="Whisper model"
            value={settings.whisper.model}
            onChange={(value) => patch({ whisper: { ...settings.whisper, model: value } })}
            options={['tiny', 'base', 'small', 'medium', 'large-v3']}
          />
          <Field
            label="Language"
            hint="Leave empty to detect automatically."
            value={settings.whisper.language}
            placeholder="auto"
            onCommit={(value) => patch({ whisper: { ...settings.whisper, language: value } })}
          />
        </div>

        <label className="mt-5 flex items-start gap-3 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={settings.whisper.diarization}
            disabled={!system?.diarization_available}
            onChange={(e) =>
              patch({ whisper: { ...settings.whisper, diarization: e.target.checked } })
            }
            className="mt-0.5 size-4 accent-sodium-500"
          />
          <span>
            Identify speakers
            {!system?.diarization_available && (
              <span className="mt-1 block text-xs text-ink-500">
                Needs the diarization extra:{' '}
                <code className="text-ink-300">uv pip install &apos;autoclip[diarization]&apos;</code>
              </span>
            )}
          </span>
        </label>
      </Section>

      <Section title="Clips">
        <div className="grid gap-5 sm:grid-cols-3">
          <NumberField
            label="Min length (s)"
            value={settings.clips.min_duration_s}
            onCommit={(value) => patch({ clips: { ...settings.clips, min_duration_s: value } })}
          />
          <NumberField
            label="Max length (s)"
            value={settings.clips.max_duration_s}
            onCommit={(value) => patch({ clips: { ...settings.clips, max_duration_s: value } })}
          />
          <NumberField
            label="Max clips"
            value={settings.clips.max_clips}
            onCommit={(value) => patch({ clips: { ...settings.clips, max_clips: value } })}
          />
        </div>
      </Section>

      <Section title="Ingest">
        <div className="grid gap-5 sm:grid-cols-2">
          <Select
            label="YouTube cookies from"
            hint="YouTube blocks most anonymous downloads. Sign in in that browser and close it before downloading."
            value={settings.ingest.cookies_from_browser}
            onChange={(value) =>
              patch({ ingest: { ...settings.ingest, cookies_from_browser: value } })
            }
            options={['', 'chrome', 'firefox', 'edge', 'brave', 'chromium', 'safari']}
            labels={{ '': 'None' }}
          />
        </div>
      </Section>

      <Section title="Export">
        <div className="grid gap-5 sm:grid-cols-2">
          <Select
            label="Default ratio"
            value={settings.export.ratio}
            onChange={(value) => patch({ export: { ...settings.export, ratio: value } })}
            options={['9:16', '1:1', '16:9']}
          />
          <NumberField
            label="Loudness target (LUFS)"
            value={settings.export.loudness_lufs}
            onCommit={(value) => patch({ export: { ...settings.export, loudness_lufs: value } })}
          />
        </div>

        <label className="mt-5 flex items-start gap-3 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={settings.export.prefer_hardware_encoder}
            onChange={(e) =>
              patch({
                export: { ...settings.export, prefer_hardware_encoder: e.target.checked },
              })
            }
            className="mt-0.5 size-4 accent-sodium-500"
          />
          <span>
            Use GPU encoding when available
            {system && !system.nvenc_works && (
              <span className="mt-1 block text-xs text-ink-500">
                Not usable on this machine — exports will use the CPU encoder. Same quality,
                slower.
              </span>
            )}
          </span>
        </label>

        <label className="mt-4 flex items-center gap-3 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={settings.export.write_srt}
            onChange={(e) => patch({ export: { ...settings.export, write_srt: e.target.checked } })}
            className="size-4 accent-sodium-500"
          />
          Also write an .srt sidecar
        </label>
      </Section>

      <Section title="Watermark" note="A logo burned into the corner of every exported clip.">
        <div className="grid gap-8 sm:grid-cols-[minmax(0,1fr)_auto]">
          <div className="space-y-6">
            <div>
              <span className="eyebrow">Image</span>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={() => watermarkFileInput.current?.click()}
                  disabled={watermarkBusy}
                  className="btn btn-ghost"
                >
                  {watermarkBusy
                    ? 'Working…'
                    : settings.export.watermark.filename
                      ? 'Replace image'
                      : 'Upload image'}
                </button>
                {settings.export.watermark.filename && (
                  <button
                    type="button"
                    onClick={removeWatermark}
                    disabled={watermarkBusy}
                    className="btn btn-quiet"
                  >
                    Remove
                  </button>
                )}
                <span className="text-xs text-ink-500">PNG (with transparency) or JPG</span>
              </div>
              <input
                ref={watermarkFileInput}
                type="file"
                className="hidden"
                accept="image/png,image/jpeg"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) void uploadWatermark(file)
                  e.target.value = ''
                }}
              />
            </div>

            <label className="flex items-start gap-3 text-sm text-ink-200">
              <input
                type="checkbox"
                checked={settings.export.watermark.enabled}
                disabled={!settings.export.watermark.filename}
                onChange={(e) => patchWatermark({ enabled: e.target.checked })}
                className="mt-0.5 size-4 accent-sodium-500"
              />
              <span>
                Stamp every export
                {!settings.export.watermark.filename && (
                  <span className="mt-1 block text-xs text-ink-500">
                    Upload an image first.
                  </span>
                )}
              </span>
            </label>

            <div>
              <span className="eyebrow">Position</span>
              <div className="mt-2 flex flex-wrap gap-2">
                {WATERMARK_POSITIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => patchWatermark({ position: option.value })}
                    className={[
                      'btn',
                      option.value === settings.export.watermark.position
                        ? 'btn-primary'
                        : 'btn-ghost',
                    ].join(' ')}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid gap-5 sm:grid-cols-2">
              <PercentSlider
                label="Size"
                hint="Scales with the exported video's width."
                value={settings.export.watermark.scale_pct}
                min={2}
                max={60}
                onCommit={(value) => patchWatermark({ scale_pct: value })}
              />
              <PercentSlider
                label="Opacity"
                value={settings.export.watermark.opacity_pct}
                min={0}
                max={100}
                onCommit={(value) => patchWatermark({ opacity_pct: value })}
              />
            </div>
          </div>

          <div>
            <span className="eyebrow">Preview</span>
            <div className="mt-2">
              <WatermarkPreview
                imageUrl={watermarkPreviewUrl}
                position={settings.export.watermark.position}
                scalePct={settings.export.watermark.scale_pct}
                opacityPct={settings.export.watermark.opacity_pct}
              />
            </div>
          </div>
        </div>
      </Section>

      {system && (
        <Section title="This machine">
          <dl className="grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2">
            <Row label="Platform" value={system.platform} />
            <Row label="Python" value={system.python_version} />
            <Row label="ffmpeg" value={system.ffmpeg_version ?? 'not found'} />
            <Row label="Acceleration" value={system.accel.toUpperCase()} />
            <Row label="Device" value={system.gpu_name ?? '—'} />
            <Row label="Whisper compute" value={system.compute_type} />
            <Row label="GPU encode" value={system.nvenc_works ? 'available' : 'unavailable'} />
            <Row label="Captions" value={system.has_libass ? 'libass present' : 'libass missing'} />
          </dl>
        </Section>
      )}
    </div>
  )
}

function Section({
  title,
  note,
  children,
}: {
  title: string
  note?: string
  children: React.ReactNode
}) {
  return (
    <section className="rise mt-14">
      <div className="border-b border-ink-800 pb-2">
        <h2 className="eyebrow">{title}</h2>
        {note && <p className="mt-1 text-xs text-ink-500">{note}</p>}
      </div>
      <div className="mt-5">{children}</div>
    </section>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-ink-850 pb-2">
      <dt className="text-ink-500">{label}</dt>
      <dd className="numeric truncate text-right text-ink-200">{value}</dd>
    </div>
  )
}

/** Commits on blur rather than per keystroke, so a PUT isn't fired per letter. */
function Field({
  label,
  hint,
  value,
  placeholder,
  onCommit,
}: {
  label: string
  hint?: string
  value: string
  placeholder?: string
  onCommit: (value: string) => void
}) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])

  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <input
        className="field mt-1 text-sm"
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => draft !== value && onCommit(draft)}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
        spellCheck={false}
      />
      {hint && <span className="mt-1.5 block text-xs leading-snug text-ink-500">{hint}</span>}
    </label>
  )
}

function NumberField({
  label,
  value,
  onCommit,
}: {
  label: string
  value: number
  onCommit: (value: number) => void
}) {
  const [draft, setDraft] = useState(String(value))
  useEffect(() => setDraft(String(value)), [value])

  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <input
        type="number"
        className="field numeric mt-1 text-sm"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          const parsed = Number(draft)
          if (!Number.isNaN(parsed) && parsed !== value) onCommit(parsed)
        }}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
      />
    </label>
  )
}

/** A range slider with a live numeric readout. Commits on release/key-up
 * rather than on every drag tick, the slider equivalent of NumberField's
 * commit-on-blur — so dragging doesn't fire a PUT per pixel. */
function PercentSlider({
  label,
  hint,
  value,
  min,
  max,
  onCommit,
}: {
  label: string
  hint?: string
  value: number
  min: number
  max: number
  onCommit: (value: number) => void
}) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])

  const commit = (next: number) => {
    const clamped = Math.min(max, Math.max(min, next))
    if (clamped !== value) onCommit(clamped)
  }

  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <div className="mt-1.5 flex items-center gap-3">
        <input
          type="range"
          min={min}
          max={max}
          value={draft}
          onChange={(e) => setDraft(Number(e.target.value))}
          onMouseUp={(e) => commit(Number(e.currentTarget.value))}
          onTouchEnd={(e) => commit(Number(e.currentTarget.value))}
          onKeyUp={(e) => commit(Number(e.currentTarget.value))}
          className="h-1 flex-1 cursor-pointer accent-sodium-500"
        />
        <span className="numeric w-10 shrink-0 text-right text-sm text-ink-300">{draft}%</span>
      </div>
      {hint && <span className="mt-1.5 block text-xs leading-snug text-ink-500">{hint}</span>}
    </label>
  )
}

/** A mock 9:16 frame with the uploaded image positioned exactly the way
 * ffmpeg's overlay filter will place it — same corner, same margin, same
 * scale-of-width, same opacity — so this is a true preview, not a mockup. */
function WatermarkPreview({
  imageUrl,
  position,
  scalePct,
  opacityPct,
}: {
  imageUrl: string | null
  position: WatermarkPosition
  scalePct: number
  opacityPct: number
}) {
  // Mirrors WATERMARK_MARGIN_FRACTION in pipeline/export.py: the gap is a
  // fraction of the frame's *width* on every side. left/right percentages
  // already mean that, but top/bottom percentages are of the height — so the
  // vertical gap is a margin instead, whose percentages always use the width.
  const margin = '4%'
  const placement: React.CSSProperties = {
    position: 'absolute',
    width: `${scalePct}%`,
    opacity: opacityPct / 100,
  }
  if (position === 'center') {
    placement.top = '50%'
    placement.left = '50%'
    placement.transform = 'translate(-50%, -50%)'
  } else {
    if (position.startsWith('top')) {
      placement.top = 0
      placement.marginTop = margin
    } else {
      placement.bottom = 0
      placement.marginBottom = margin
    }
    if (position.endsWith('left')) placement.left = margin
    else placement.right = margin
  }

  return (
    <div className="relative aspect-[9/16] w-36 overflow-hidden rounded-[var(--radius-field)] border border-ink-800 bg-gradient-to-br from-ink-850 to-ink-900">
      <span className="absolute inset-0 grid place-items-center px-3 text-center text-[0.65rem] leading-snug text-ink-700">
        9:16 frame
      </span>
      {imageUrl && (
        <img src={imageUrl} alt="Watermark placement preview" style={placement} className="pointer-events-none" />
      )}
    </div>
  )
}

function Select({
  label,
  hint,
  value,
  onChange,
  options,
  labels = {},
}: {
  label: string
  hint?: string
  value: string
  onChange: (value: string) => void
  options: string[]
  labels?: Record<string, string>
}) {
  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <select
        className="field mt-1 cursor-pointer text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((option) => (
          <option key={option} value={option} className="bg-ink-850">
            {labels[option] ?? option}
          </option>
        ))}
      </select>
      {hint && <span className="mt-1.5 block text-xs leading-snug text-ink-500">{hint}</span>}
    </label>
  )
}

function SecretField({
  secretKey,
  label,
  present,
  onChanged,
  onError,
}: {
  secretKey: string
  label: string
  present: boolean
  onChanged: () => void
  onError: (error: Error) => void
}) {
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState(false)

  const save = async () => {
    if (!value.trim()) return
    setBusy(true)
    try {
      await api.putSecret(secretKey, value.trim())
      setValue('')
      onChanged()
    } catch (err) {
      onError(err as Error)
    } finally {
      setBusy(false)
    }
  }

  const remove = async () => {
    setBusy(true)
    try {
      await api.deleteSecret(secretKey)
      onChanged()
    } catch (err) {
      onError(err as Error)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-wrap items-end gap-3">
      <label className="min-w-56 flex-1">
        <span className="eyebrow">
          {label}
          {present && <span className="ml-2 text-signal-good">set</span>}
        </span>
        <input
          type="password"
          className="field mt-1 text-sm"
          value={value}
          placeholder={present ? '••••••••••••' : 'paste to add'}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && save()}
          autoComplete="off"
        />
      </label>
      <button onClick={save} disabled={!value.trim() || busy} className="btn btn-ghost">
        Save
      </button>
      {present && (
        <button onClick={remove} disabled={busy} className="btn btn-quiet">
          Remove
        </button>
      )}
    </div>
  )
}
