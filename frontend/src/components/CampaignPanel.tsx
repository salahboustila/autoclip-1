import { useEffect, useState } from 'react'

import {
  api,
  formatDuration,
  type CampaignEpisode,
  type CampaignPresetDetail,
  type CampaignPresetInfo,
} from '../api'

/**
 * Podcast Campaign Mode on the New page: a switch, the preset, its rules, and
 * a picker for the episodes the campaign accepts.
 */
export function CampaignPanel({
  presets,
  value,
  onChange,
  onPickEpisode,
}: {
  presets: CampaignPresetInfo[]
  /** The selected preset key, or undefined when the mode is off. */
  value: string | undefined
  onChange: (key: string | undefined) => void
  onPickEpisode: (url: string) => void
}) {
  const on = value !== undefined
  const [detail, setDetail] = useState<CampaignPresetDetail | null>(null)
  const [episodes, setEpisodes] = useState<CampaignEpisode[] | null>(null)
  const [loadingEpisodes, setLoadingEpisodes] = useState(false)
  const [episodeError, setEpisodeError] = useState<string | null>(null)

  useEffect(() => {
    setEpisodes(null)
    setEpisodeError(null)
    if (!value) {
      setDetail(null)
      return
    }
    api
      .getCampaign(value)
      .then(setDetail)
      .catch(() => setDetail(null))
  }, [value])

  const loadEpisodes = async () => {
    if (!value) return
    setLoadingEpisodes(true)
    setEpisodeError(null)
    try {
      setEpisodes(await api.campaignEpisodes(value))
    } catch (err) {
      setEpisodeError((err as Error).message)
    } finally {
      setLoadingEpisodes(false)
    }
  }

  if (presets.length === 0) return null

  return (
    <div
      className={[
        'mt-6 border px-4 py-3 transition-colors duration-200',
        on ? 'border-sodium-500 bg-ink-850/60' : 'border-ink-800',
      ].join(' ')}
    >
      <label className="flex cursor-pointer items-start gap-3">
        <input
          type="checkbox"
          role="switch"
          aria-checked={on}
          checked={on}
          onChange={(e) => onChange(e.target.checked ? presets[0].key : undefined)}
          className="mt-1 size-4 accent-sodium-500"
        />
        <span>
          <span className="text-sm font-medium text-ink-100">Podcast Campaign Mode</span>
          <span className="mt-1 block text-xs leading-relaxed text-ink-500">
            Clips for a clipping campaign: every clip opens on the host&apos;s question, only fresh
            moments, the campaign&apos;s tags in the caption, no watermark. Includes Viral Hook
            Mode.
          </span>
        </span>
      </label>

      {on && (
        <div className="mt-4 space-y-4 pl-7">
          <label className="block">
            <span className="eyebrow">Preset</span>
            <select
              className="field mt-1 cursor-pointer text-sm"
              value={value}
              onChange={(e) => onChange(e.target.value)}
            >
              {presets.map((preset) => (
                <option key={preset.key} value={preset.key} className="bg-ink-850">
                  {preset.name}
                  {preset.platform ? ` — ${preset.platform}` : ''}
                </option>
              ))}
            </select>
          </label>

          {detail && (
            <ul className="space-y-1 text-xs leading-relaxed text-ink-400">
              <li>
                Only {detail.host}&apos;s latest {detail.latest_episodes} episodes.
                {detail.allow_uploads && ' Uploaded files are marked “source not verified”.'}
              </li>
              <li>
                {detail.min_duration_s}–{detail.max_duration_s} s, top {detail.top_n}, score ≥{' '}
                {detail.min_score}. Opens on {detail.host}&apos;s question.
              </li>
              <li>Caption tags: {detail.caption_lines.join(' ')}</li>
            </ul>
          )}

          <div>
            <button
              type="button"
              onClick={() => void loadEpisodes()}
              disabled={loadingEpisodes}
              className="btn btn-ghost text-xs"
            >
              {loadingEpisodes
                ? 'Loading episodes…'
                : episodes
                  ? 'Refresh episodes'
                  : `Pick from the latest ${detail?.latest_episodes ?? ''} episodes`}
            </button>
            {episodeError && <p className="mt-2 text-xs text-signal-bad">{episodeError}</p>}
            {episodes && (
              <ol className="mt-3 max-h-72 space-y-1 overflow-y-auto pr-1">
                {episodes.map((episode, index) => (
                  <li key={episode.id}>
                    <button
                      type="button"
                      onClick={() => onPickEpisode(episode.url)}
                      className="grid w-full grid-cols-[1.75rem_1fr_auto] items-baseline gap-2 py-1 text-left text-xs text-ink-300 transition-colors hover:text-ink-100"
                    >
                      <span className="numeric text-ink-600">{index + 1}</span>
                      <span className="truncate">{episode.title}</span>
                      <span className="numeric text-ink-600">
                        {episode.duration_s ? formatDuration(episode.duration_s) : ''}
                      </span>
                    </button>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
