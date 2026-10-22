import { createContext, useContext, useEffect, useState, useCallback } from 'react'
import { api, setActiveDataset } from '../services/api'

const DatasetContext = createContext(null)
const STORAGE_KEY = 'cip.dataset'

function initialDataset() {
  try { return localStorage.getItem(STORAGE_KEY) || 'synthetic' } catch { return 'synthetic' }
}

export function DatasetProvider({ children }) {
  const [dataset, setDatasetState] = useState(initialDataset)
  const [datasets, setDatasets] = useState([])

  // Set during render so child effects and requests already use the new dataset
  setActiveDataset(dataset)

  const refresh = useCallback(async () => {
    try { setDatasets(await api.getDatasets()) } catch { /* backend offline */ }
  }, [])

  useEffect(() => { refresh() }, [refresh, dataset])

  const setDataset = (id) => {
    try { localStorage.setItem(STORAGE_KEY, id) } catch { /* storage blocked */ }
    setDatasetState(id)
  }

  const current = datasets.find((d) => d.id === dataset) ?? null
  return (
    <DatasetContext.Provider value={{ dataset, setDataset, datasets, current, refresh }}>
      {children}
    </DatasetContext.Provider>
  )
}

export const useDataset = () => useContext(DatasetContext)
