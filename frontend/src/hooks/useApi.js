import { useState, useEffect, useCallback } from 'react'

export function useApi(apiFn, deps = []) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetch = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await apiFn()
      setData(result)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, deps) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { fetch() }, [fetch])

  return { data, loading, error, refetch: fetch }
}

export function usePoll(apiFn, intervalMs = 3000, condition = null) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    const run = async () => {
      try {
        const result = await apiFn()
        if (active) setData(result)
        setLoading(false)
        if (condition && condition(result)) return
        if (active) setTimeout(run, intervalMs)
      } catch {
        if (active) setTimeout(run, intervalMs)
      }
    }
    run()
    return () => { active = false }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return { data, loading }
}
