import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/useAuth'

export function Navbar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <nav className="border-b border-neutral-200 bg-white">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3">
        <Link to="/" className="font-semibold tracking-tight text-neutral-900">
          fl-web-service
        </Link>
        <div className="flex items-center gap-4 text-sm">
          {user ? (
            <>
              <Link to="/tokens" className="text-neutral-700 hover:text-neutral-900">
                Client tokens
              </Link>
              <span className="text-neutral-500">{user.email}</span>
              <button
                type="button"
                onClick={handleLogout}
                className="rounded border border-neutral-300 px-3 py-1 text-neutral-700 hover:bg-neutral-100"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="text-neutral-700 hover:text-neutral-900">
                Login
              </Link>
              <Link
                to="/register"
                className="rounded bg-neutral-900 px-3 py-1 text-white hover:bg-neutral-700"
              >
                Register
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  )
}
