import axios from 'axios'

const BASE = '/api'

const client = axios.create({
  baseURL: BASE,
  timeout: 300000, // 5 min for long-running ops
})

client.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const msg = err.response?.data?.detail || err.message || 'Unknown error'
    console.error('[API Error]', msg)
    return Promise.reject(new Error(msg))
  }
)

export const api = {
  health: () => client.get('/health'),

  // Data
  generateData: (nRows) => client.post(`/generate-data${nRows ? `?n_rows=${nRows}` : ''}`),
  getDataStatus: () => client.get('/generate-data/status'),
  getDataInfo: () => client.get('/generate-data/info'),

  // Training
  trainModels: (nClusters, method) =>
    client.post(`/train-models?n_clusters=${nClusters || 6}&method=${method || 'kmeans'}`),
  getTrainStatus: () => client.get('/train-models/status'),

  // Results
  getSegments: () => client.get('/segments'),
  getSegmentDetail: (id) => client.get(`/segments/${id}`),
  getPersonas: () => client.get('/personas'),
  getPredictions: () => client.get('/predictions'),
  getExplanations: () => client.get('/explanations'),
  getBiasReport: () => client.get('/bias-report'),

  // What-if
  whatIf: (features) => client.post('/predictions/what-if', { features }),
}
