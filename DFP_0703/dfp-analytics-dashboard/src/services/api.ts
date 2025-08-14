import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
})

export interface DataSummary {
  total_records: number
  date_range: {
    start: string | null
    end: string | null
  }
  unique_counts: {
    regions: number
    markets: number
    locations: number
    form_types: number
    users: number
  }
  columns: string[]
}

export interface FilterOptions {
  regions: string[]
  markets: string[]
  form_types: string[]
  tmos: string[]
  years: number[]
  months: number[]
}

export interface PaginatedRecords {
  records: any[]
  pagination: {
    page: number
    limit: number
    total_records: number
    total_pages: number
  }
}

export interface GeographicAnalytics {
  region_distribution: Record<string, number>
  market_distribution: Record<string, number>
  location_hierarchy: Array<{
    region: string
    market: string
    count: number
  }>
}

export interface TemporalAnalytics {
  daily_trends: Array<{
    date: string
    count: number
  }>
  monthly_by_form_type: Array<{
    month: string
    form_type: string
    count: number
  }>
  weekday_distribution: Record<string, number>
}

export interface BrandAnalytics {
  brand_mentions: Record<string, number>
  tmo_distribution: Record<string, number>
  brand_by_region: Array<{
    region: string
    brand: string
    count: number
  }>
}

export interface FormAnalytics {
  form_distribution: Record<string, number>
  form_by_region: Array<{
    region: string
    form_type: string
    count: number
  }>
  transaction_statistics: {
    mean: number
    median: number
    std: number
    total: number
  }
}

export interface UserAnalytics {
  top_contributors: Record<string, number>
  users_by_region: Array<{
    region: string
    unique_users: number
  }>
  user_activity_timeline: Array<{
    month: string
    active_users: number
  }>
}

// API Functions
export const fetchDataSummary = async (): Promise<DataSummary> => {
  const response = await apiClient.get('/data/summary')
  return response.data
}

export const fetchFilterOptions = async (): Promise<FilterOptions> => {
  const response = await apiClient.get('/data/filters')
  return response.data
}

export const fetchFilteredRecords = async (
  page: number = 1,
  limit: number = 100,
  filters: Record<string, any> = {}
): Promise<PaginatedRecords> => {
  const params = new URLSearchParams({
    page: page.toString(),
    limit: limit.toString(),
  })

  Object.entries(filters).forEach(([key, value]) => {
    if (value && value !== '') {
      params.append(key, value.toString())
    }
  })

  const response = await apiClient.get(`/data/records?${params}`)
  return response.data
}

export const fetchGeographicAnalytics = async (): Promise<GeographicAnalytics> => {
  const response = await apiClient.get('/analytics/geographic')
  return response.data
}

export const fetchTemporalAnalytics = async (): Promise<TemporalAnalytics> => {
  const response = await apiClient.get('/analytics/temporal')
  return response.data
}

export const fetchBrandAnalytics = async (): Promise<BrandAnalytics> => {
  const response = await apiClient.get('/analytics/brands')
  return response.data
}

export const fetchFormAnalytics = async (): Promise<FormAnalytics> => {
  const response = await apiClient.get('/analytics/forms')
  return response.data
}

export const fetchUserAnalytics = async (): Promise<UserAnalytics> => {
  const response = await apiClient.get('/analytics/users')
  return response.data
}

export const exportFilteredData = async (filters: Record<string, any> = {}): Promise<void> => {
  const params = new URLSearchParams()
  
  Object.entries(filters).forEach(([key, value]) => {
    if (value && value !== '') {
      params.append(key, value.toString())
    }
  })

  const response = await apiClient.get(`/export/csv?${params}`, {
    responseType: 'blob',
  })

  // Create download link
  const url = window.URL.createObjectURL(new Blob([response.data]))
  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', 'dfp_filtered_data.csv')
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.URL.revokeObjectURL(url)
} 