import { useEffect, useState } from 'react'

import type { Clip } from '../api'
import { HookTitleField } from './HookTitleField'

type ClipPatch = { hook_title?: string; hook_title_alts?: string[]; post_caption?: string }

/**
 * Hook title, its alternatives, the post caption, and a re-render button.
 *
 * Alternatives swap rather than replace: picking one moves the current title
 * into the alternatives, so nothing the model wrote is ever lost by a click.
 */
export function HookCopyPanel({
  clip,
  onPatch,
  onRerender,
  rerendering,
  rerenderBlocked,
}: {
  clip: Clip
  onPatch: (patch: ClipPatch) => Promise<void>
  onRerender: () => Promise<void>
  rerendering: boolean
  /** Why re-rendering is impossible (e.g. the source was deleted), or null. */
  rerenderBlocked: string | null
}) {
  const pickAlternative = (alt: string) => {
    const others = clip.hook_title_alts.filter((t) => t !== alt)
    const alts = clip.hook_title ? [clip.hook_title, ...others] : others
    return onPatch({ hook_title: alt, hook_title_alts: alts })
  }

  const exported = clip.exports.length > 0

  return (
    <div className="space-y-6">
      {(clip.topic || clip.score > 0) && (
        <div className="flex items-baseline gap-3">
          <span className="numeric font-display text-2xl leading-none text-sodium-500">
            {clip.score}
          </span>
          {clip.topic && (
            <span className="border border-ink-700 px-2 py-0.5 text-xs uppercase tracking-wide text-ink-300">
              {clip.topic}
            </span>
          )}
        </div>
      )}

      {clip.question_text && (
        <div>
          <p className="eyebrow">Opens on the question</p>
          <p className="mt-2 border-l-2 border-sodium-500 pl-3 text-[0.9375rem] leading-relaxed text-ink-100">
            {clip.question_text}
          </p>
        </div>
      )}

      <HookTitleField
        value={clip.hook_title}
        onSave={(hookTitle) => onPatch({ hook_title: hookTitle })}
      />

      {clip.hook_title_alts.length > 0 && (
        <div>
          <p className="eyebrow">Alternatives</p>
          <ul className="mt-2 space-y-1">
            {clip.hook_title_alts.map((alt) => (
              <li key={alt} className="flex items-baseline justify-between gap-3">
                <span className="text-sm text-ink-200">{alt}</span>
                <button
                  type="button"
                  onClick={() => void pickAlternative(alt)}
                  className="btn btn-quiet shrink-0 text-xs"
                >
                  Use this
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <CaptionField
        value={clip.post_caption}
        onSave={(caption) => onPatch({ post_caption: caption })}
      />

      <div>
        <button
          type="button"
          onClick={() => void onRerender()}
          disabled={rerendering || rerenderBlocked !== null || !clip.hook_title.trim()}
          className="btn btn-ghost w-full"
        >
          {rerendering
            ? 'Rendering…'
            : exported
              ? 'Re-render with edited title'
              : 'Render with this title'}
        </button>
        {rerenderBlocked && <p className="mt-1.5 text-xs text-ink-500">{rerenderBlocked}</p>}
      </div>
    </div>
  )
}

function CaptionField({
  value,
  onSave,
}: {
  value: string
  onSave: (next: string) => Promise<void>
}) {
  const [draft, setDraft] = useState(value)
  const [saving, setSaving] = useState(false)
  const [copied, setCopied] = useState(false)

  useEffect(() => setDraft(value), [value])

  const dirty = draft.trim() !== value.trim()

  const save = async () => {
    if (!dirty || saving) return
    setSaving(true)
    try {
      await onSave(draft.trim())
    } finally {
      setSaving(false)
    }
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(draft.trim())
    } catch {
      // Clipboard API needs a secure context; fall back for plain-http LAN use.
      const area = document.createElement('textarea')
      area.value = draft.trim()
      document.body.appendChild(area)
      area.select()
      document.execCommand('copy')
      area.remove()
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div>
      <div className="flex items-baseline justify-between border-b border-ink-800 pb-2">
        <label htmlFor="post-caption" className="eyebrow">
          Post caption
        </label>
        <button
          type="button"
          onClick={() => void copy()}
          disabled={!draft.trim()}
          className="btn btn-quiet text-xs"
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <textarea
        id="post-caption"
        value={draft}
        rows={Math.max(4, draft.split('\n').length + 1)}
        maxLength={2200}
        placeholder="No caption yet"
        onChange={(e) => setDraft(e.target.value)}
        className="mt-3 w-full resize-y border border-ink-700 bg-ink-850 px-3 py-2 text-sm leading-relaxed text-ink-100 placeholder:text-ink-600 focus:border-sodium-500 focus:outline-none"
      />
      {dirty && (
        <div className="mt-1.5 flex gap-3">
          <button
            type="button"
            onClick={() => void save()}
            disabled={saving}
            className="btn btn-primary text-xs"
          >
            {saving ? 'Saving…' : 'Save caption'}
          </button>
          <button type="button" onClick={() => setDraft(value)} className="btn btn-quiet text-xs">
            Undo
          </button>
        </div>
      )}
    </div>
  )
}
