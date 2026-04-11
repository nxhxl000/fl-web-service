import { useEffect, useState, type FormEvent } from 'react'
import {
  createClientToken,
  deleteClientToken,
  listClientTokens,
} from '../api/clientTokens'
import { ApiError } from '../api/client'
import type { ClientToken } from '../api/types'

export function TokensPage() {
  const [tokens, setTokens] = useState<ClientToken[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const [justCreated, setJustCreated] = useState<{ id: number; token: string } | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await listClientTokens()
      setTokens(data)
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'Failed to load tokens')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const handleCreate = async (e: FormEvent) => {
    e.preventDefault()
    if (!newName.trim()) return
    setCreating(true)
    setError(null)
    try {
      const created = await createClientToken(newName.trim())
      setJustCreated({ id: created.id, token: created.token })
      setNewName('')
      await load()
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'Failed to create token')
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this token? Clients using it will stop working.')) return
    try {
      await deleteClientToken(id)
      if (justCreated?.id === id) setJustCreated(null)
      await load()
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'Failed to delete token')
    }
  }

  return (
    <main className="mx-auto max-w-3xl px-6 py-10">
      <h1 className="text-2xl font-semibold tracking-tight text-neutral-900">Client tokens</h1>
      <p className="mt-2 text-sm text-neutral-600">
        Create opaque tokens to pass into the federated learning client container as{' '}
        <code className="rounded bg-neutral-100 px-1 py-0.5 text-xs">FL_TOKEN</code>.
      </p>

      <form onSubmit={handleCreate} className="mt-6 flex gap-2">
        <input
          type="text"
          placeholder="Token name, e.g. 'laptop' or 'vm-client1'"
          required
          maxLength={100}
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          className="flex-1 rounded border border-neutral-300 px-3 py-2 text-sm focus:border-neutral-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={creating}
          className="rounded bg-neutral-900 px-4 py-2 text-sm text-white hover:bg-neutral-700 disabled:bg-neutral-400"
        >
          {creating ? 'Creating…' : 'Create'}
        </button>
      </form>

      {justCreated && (
        <div className="mt-4 rounded border border-amber-300 bg-amber-50 p-4">
          <p className="text-sm font-medium text-amber-900">
            This is the only time the token is shown. Copy it now.
          </p>
          <div className="mt-2 flex items-center gap-2">
            <code className="flex-1 break-all rounded bg-white px-3 py-2 font-mono text-xs text-neutral-900">
              {justCreated.token}
            </code>
            <button
              type="button"
              onClick={() => navigator.clipboard.writeText(justCreated.token)}
              className="rounded border border-neutral-300 px-3 py-2 text-xs text-neutral-700 hover:bg-neutral-100"
            >
              Copy
            </button>
          </div>
        </div>
      )}

      {error && <p className="mt-4 text-sm text-red-600">{error}</p>}

      <div className="mt-8">
        {loading ? (
          <p className="text-sm text-neutral-500">Loading…</p>
        ) : tokens.length === 0 ? (
          <p className="text-sm text-neutral-500">No tokens yet.</p>
        ) : (
          <ul className="divide-y divide-neutral-200 rounded border border-neutral-200 bg-white">
            {tokens.map((t) => (
              <li key={t.id} className="flex items-center justify-between px-4 py-3">
                <div>
                  <div className="text-sm font-medium text-neutral-900">{t.name}</div>
                  <div className="text-xs text-neutral-500">
                    Created {new Date(t.created_at).toLocaleString()}
                    {t.last_seen_at
                      ? ` · last seen ${new Date(t.last_seen_at).toLocaleString()}`
                      : ' · never seen'}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => handleDelete(t.id)}
                  className="rounded border border-red-300 px-3 py-1 text-xs text-red-700 hover:bg-red-50"
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </main>
  )
}
