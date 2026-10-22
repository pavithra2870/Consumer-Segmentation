import axios from 'axios'

const BASE = '/api'

const client = axios.create({
  baseURL: BASE,
  timeout: 300000, // 5 min for long-running ops
})

// Every request is scoped to the dataset currently selected in the sidebar
let activeDataset = 'synthetic'
export const setActiveDataset = (id) => { activeDataset = id }

client.interceptors.request.use((config) => {
  config.params = { dataset: activeDataset, ...config.params }
  return config
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
  getDatasets: () => client.get('/datasets'),

  // Data
  generateData: (nRows) => client.post('/generate-data', null, { params: nRows ? { n_rows: nRows } : {} }),
  getDataStatus: () => client.get('/generate-data/status'),
  getDataInfo: () => client.get('/generate-data/info'),

  // Training
  trainModels: (nClusters, method) =>
    client.post('/train-models', null, { params: { n_clusters: nClusters || 6, method: method || 'kmeans' } }),
  getTrainStatus: () => client.get('/train-models/status'),

  // Results
  getSegments: () => client.get('/segments'),
  getSegmentDetail: (id) => client.get(`/segments/${id}`),
  getPersonas: () => client.get('/personas'),
  getPredictions: () => client.get('/predictions'),
  getExplanations: () => client.get('/explanations'),
  getBiasReport: () => client.get('/bias-report'),

  // Benchmarks
  getBenchmarkDatasets: () => client.get('/benchmarks/datasets'),
  getBenchmark: (id) => client.get(`/benchmarks/${id}`),
  runBenchmarks: (id) => client.post('/benchmarks/run', null, { params: { dataset: id || 'all' } }),
  getBenchmarkStatus: () => client.get('/benchmarks/status'),

  // What-if
  getWhatIfSchema: () => client.get('/predictions/what-if/schema'),
  whatIf: (features) => client.post('/predictions/what-if', { features }),
}
