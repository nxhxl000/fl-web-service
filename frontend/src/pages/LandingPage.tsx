import { Link } from 'react-router-dom'
import { useAuth } from '../auth/useAuth'

export function LandingPage() {
  const { user } = useAuth()

  return (
    <main className="mx-auto max-w-2xl px-6 py-16 text-center">
      <h1 className="text-4xl font-semibold tracking-tight text-neutral-900">fl-web-service</h1>
      <p className="mt-4 text-neutral-600">
        Web service layer for the federated plant disease classification thesis project.
      </p>
      <div className="mt-8 flex justify-center gap-3">
        {user ? (
          <Link
            to="/tokens"
            className="rounded bg-neutral-900 px-4 py-2 text-sm text-white hover:bg-neutral-700"
          >
            Go to client tokens
          </Link>
        ) : (
          <>
            <Link
              to="/login"
              className="rounded border border-neutral-300 px-4 py-2 text-sm text-neutral-800 hover:bg-neutral-100"
            >
              Login
            </Link>
            <Link
              to="/register"
              className="rounded bg-neutral-900 px-4 py-2 text-sm text-white hover:bg-neutral-700"
            >
              Register
            </Link>
          </>
        )}
      </div>
    </main>
  )
}
