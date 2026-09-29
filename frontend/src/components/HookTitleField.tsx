import { useEffect, useState } from 'react'

/**
 * Editable hook title — the headline burned over the whole clip.
 *
 * Saved on Enter or blur rather than per keystroke, so a half-typed title
 * never lands in the database. Empty clears the overlay.
 */
export function HookTitleField({
  value,
  onSave,
}: {
  value: string
  onSave: (next: string) => Promise<void>
}) {
  const [draft, setDraft] = useState(value)
  const [saving, setSaving] = useState(false)

  useEffect(() => setDraft(value), [value])

  const dirty = draft.trim() !== value.trim()

  const commit = async () => {
    if (!dirty || saving) return
    setSaving(true)
    try {
      await onSave(draft.trim())
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <label htmlFor="hook-title" className="eyebrow block border-b border-ink-800 pb-2">
        Hook title
      </label>
      <input
        id="hook-title"
        value={draft}
        maxLength={200}
        placeholder="No title overlay"
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') void commit()
          if (e.key === 'Escape') setDraft(value)
        }}
        className="mt-3 w-full border border-ink-700 bg-ink-850 px-3 py-2 text-sm text-ink-100 placeholder:text-ink-600 focus:border-sodium-500 focus:outline-none"
      />
      <p className="mt-1.5 text-xs text-ink-500">
        {saving
          ? 'Saving…'
          : dirty
            ? 'Enter to save · Esc to undo'
            : 'Shown at the top for the whole clip. Max two lines; emoji supported.'}
      </p>
    </div>
  )
}
