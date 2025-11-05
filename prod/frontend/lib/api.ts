/**
 * Client API Oddsy avec types TypeScript stricts
 * =============================================
 * Interface vers backend FastAPI avec validation Zod
 */

import { z } from 'zod';
import type {
  APIResponse,
  RoundPredictions,
  PipelineStatus,
  JobStatusResponse,
  OddsyAPIConfig,
  FetchOptions
} from './types';
import { fetchGameweekPredictions } from './api-v5-adapter';

// ===== ZOD SCHEMAS VALIDATION =====

const PredictionProbabilitiesSchema = z.object({
  home: z.number().min(0).max(1),
  draw: z.number().min(0).max(1),
  away: z.number().min(0).max(1)
}).refine(
  (data) => {
    const total = data.home + data.draw + data.away;
    return total >= 0.98 && total <= 1.02;
  },
  { message: "Probabilities must sum to approximately 1.0" }
);

const ModelPredictionSchema = z.object({
  prediction: z.enum(["H", "D", "A"]),
  confidence: z.number().min(0).max(1),
  probabilities: PredictionProbabilitiesSchema
});

const MatchPredictionSchema = z.object({
  id: z.string(),
  home_team: z.string(),
  away_team: z.string(),
  date: z.string(),
  round: z.number().min(1).max(38),
  ensemble: ModelPredictionSchema,
  models: z.record(ModelPredictionSchema),
  disagreement: z.number().min(0).max(1)
});

const RoundPredictionsSchema = z.object({
  round: z.number().min(1).max(38),
  season: z.string().regex(/^\d{4}-\d{2}$/),
  competition: z.string(),
  total_matches: z.number().min(1).max(20), // Allow up to 20 for multi-GW
  ensemble_system: z.object({
    system_name: z.string(),
    version: z.string(),
    models: z.record(z.any()),
    ensemble_strategy: z.string(),
    weights: z.record(z.number()),
    expected_performance: z.number().min(0).max(1)
  }),
  matches: z.array(MatchPredictionSchema).min(1).max(20) // Allow up to 20 for multi-GW
});

const APIResponseSchema = <T extends z.ZodType>(dataSchema: T) => z.object({
  meta: z.object({
    api_version: z.string(),
    pipeline_version: z.string(),
    generated_at: z.string(),
    git_sha: z.string().optional()
  }),
  data: dataSchema,
  error: z.string().optional()
});

// ===== API CLIENT CLASS =====

export class OddsyAPIClient {
  private baseURL: string;
  private timeout: number;
  private apiVersion: string;

  constructor(config: OddsyAPIConfig) {
    this.baseURL = config.baseURL.replace(/\/$/, ''); // Remove trailing slash
    this.timeout = config.timeout || 10000; // 10s default
    this.apiVersion = config.apiVersion || '1.0';
  }

  /**
   * Generic fetch with timeout and error handling
   */
  private async fetchWithTimeout<T>(
    endpoint: string, 
    options: RequestInit & FetchOptions = {}
  ): Promise<T> {
    const { cache, revalidate, ...fetchOptions } = options;
    
    const url = `${this.baseURL}${endpoint}`;
    
    // Setup fetch options
    const requestOptions: RequestInit = {
      ...fetchOptions,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-API-Version': this.apiVersion,
        ...fetchOptions.headers
      }
    };

    // Add cache options for Next.js
    if (cache) {
      (requestOptions as any).cache = cache;
    }
    if (revalidate !== undefined) {
      (requestOptions as any).next = { revalidate };
    }

    // Create timeout controller
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);
    
    try {
      const response = await fetch(url, {
        ...requestOptions,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorData}`);
      }

      const data = await response.json();
      return data as T;

    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error(`Request timeout after ${this.timeout}ms`);
        }
        throw error;
      }
      
      throw new Error('Unknown fetch error');
    }
  }

  /**
   * Validated fetch with Zod schema
   */
  private async fetchValidated<T>(
    endpoint: string,
    schema: z.ZodType<T>,
    options: RequestInit & FetchOptions = {}
  ): Promise<T> {
    const data = await this.fetchWithTimeout<T>(endpoint, options);
    
    try {
      return schema.parse(data);
    } catch (error) {
      if (error instanceof z.ZodError) {
        console.error('API Response validation failed:', error.issues);
        throw new Error(`Invalid API response: ${error.issues.map(i => i.message).join(', ')}`);
      }
      throw error;
    }
  }

  // ===== PREDICTIONS ENDPOINTS =====

  /**
   * Récupère prédictions d'une journée (via adaptateur v5)
   */
  async getPredictions(
    round: number, 
    options: FetchOptions & {
      date_filter?: string;
      allow_multi_gw?: boolean;
    } = {}
  ): Promise<APIResponse<RoundPredictions>> {
    if (round < 1 || round > 38) {
      throw new Error(`Invalid round: ${round}. Must be between 1 and 38`);
    }

    // Use API v5 adapter for now
    try {
      return await fetchGameweekPredictions(round, this.baseURL);
    } catch (error) {
      console.error(`Failed to fetch predictions for round ${round}:`, error);
      throw error;
    }
  }

  /**
   * Liste journées disponibles
   */
  async getAvailableRounds(options: FetchOptions = {}): Promise<APIResponse<{
    available_rounds: Array<{
      round: number;
      file_name: string;
      generated_at: string;
      file_size_kb: number;
    }>;
    total_rounds: number;
    latest_round: number;
  }>> {
    const response = await this.fetchWithTimeout(
      '/api/v5/gameweeks/available',
      {
        method: 'GET',
        cache: 'default',
        revalidate: 300, // 5min cache
        ...options
      }
    );

    // Adapter la réponse API v5 vers le format frontend
    const v5Data = response.data;
    
    return {
      meta: response.meta,
      data: {
        available_rounds: v5Data.available_gameweeks.map((gw: any) => ({
          round: gw.gameweek,
          file_name: gw.directory || `gw${gw.gameweek}`,
          generated_at: gw.metadata?.generated_at || new Date().toISOString(),
          file_size_kb: 0 // Non disponible dans API v5
        })),
        total_rounds: v5Data.statistics.total_gameweeks,
        latest_round: v5Data.statistics.latest_gameweek
      }
    };
  }

  // ===== PIPELINE ENDPOINTS =====

  /**
   * Status Pipeline Durci
   */
  async getPipelineStatus(options: FetchOptions = {}): Promise<APIResponse<PipelineStatus>> {
    return this.fetchWithTimeout(
      '/api/v1/pipeline/status',
      {
        method: 'GET',
        cache: 'no-store', // Always fresh
        ...options
      }
    );
  }

  /**
   * Déclenche génération journée
   */
  async triggerRoundGeneration(round: number): Promise<APIResponse<{
    job_id: string;
    round: number;
    status: string;
    message: string;
    check_status_url: string;
  }>> {
    if (round < 8 || round > 38) {
      throw new Error(`Invalid round for generation: ${round}. Only J8-J38 can be triggered`);
    }

    return this.fetchWithTimeout(
      `/api/v1/pipeline/trigger/j${round}`,
      {
        method: 'POST',
        cache: 'no-store'
      }
    );
  }

  /**
   * Status d'un job
   */
  async getJobStatus(jobId: string, options: FetchOptions = {}): Promise<APIResponse<JobStatusResponse>> {
    return this.fetchWithTimeout(
      `/api/v1/pipeline/jobs/${jobId}`,
      {
        method: 'GET',
        cache: 'no-store',
        ...options
      }
    );
  }

  /**
   * Liste des jobs
   */
  async getJobs(limit: number = 10, options: FetchOptions = {}): Promise<APIResponse<{
    jobs: JobStatusResponse[];
    total_displayed: number;
    limit: number;
  }>> {
    return this.fetchWithTimeout(
      `/api/v1/pipeline/jobs?limit=${limit}`,
      {
        method: 'GET',
        cache: 'default',
        revalidate: 60, // 1min cache
        ...options
      }
    );
  }

  // ===== HEALTH ENDPOINTS =====

  /**
   * Health check
   */
  async getHealth(): Promise<{
    status: string;
    timestamp: string;
    service: string;
  }> {
    return this.fetchWithTimeout('/api/v1/health/live', {
      method: 'GET',
      cache: 'no-store'
    });
  }

  /**
   * Readiness check
   */
  async getReadiness(): Promise<{
    status: string;
    timestamp: string;
    checks: Record<string, any>;
  }> {
    return this.fetchWithTimeout('/api/v1/health/ready', {
      method: 'GET',
      cache: 'no-store'
    });
  }

  /**
   * Métriques système
   */
  async getMetrics(): Promise<{
    timestamp: string;
    version: Record<string, string>;
    system: Record<string, number>;
    pipeline: Record<string, any>;
    settings: Record<string, any>;
  }> {
    return this.fetchWithTimeout('/api/v1/health/metrics', {
      method: 'GET',
      cache: 'default',
      revalidate: 300 // 5min cache
    });
  }
}

// ===== DEFAULT CLIENT INSTANCE =====

export const createOddsyAPI = (config?: Partial<OddsyAPIConfig>): OddsyAPIClient => {
  const defaultConfig: OddsyAPIConfig = {
    baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    timeout: 10000,
    apiVersion: '1.0'
  };

  return new OddsyAPIClient({ ...defaultConfig, ...config });
};

// Default instance
export const oddsyAPI = createOddsyAPI();

// ===== ERROR HANDLING HELPERS =====

export class OddsyAPIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public endpoint?: string
  ) {
    super(message);
    this.name = 'OddsyAPIError';
  }
}

export function isOddsyAPIError(error: unknown): error is OddsyAPIError {
  return error instanceof OddsyAPIError;
}

// ===== CACHE HELPERS =====

export const getCacheKey = (endpoint: string, params?: Record<string, any>) => {
  const paramString = params ? `?${new URLSearchParams(params).toString()}` : '';
  return `oddsy-api:${endpoint}${paramString}`;
};

export const shouldRevalidate = (round: number): boolean => {
  // Always revalidate current/future rounds
  return round <= 7;
};