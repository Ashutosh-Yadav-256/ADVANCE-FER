import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface FaceDetection {
  face_id: number;
  bbox: number[];
  dominant_emotion: string;
  confidence: number;
  probabilities: Record<string, number>;
  valence: number;
  arousal: number;
}

export interface PredictResponse {
  success: boolean;
  image_width: number;
  image_height: number;
  faces_detected: number;
  faces: FaceDetection[];
  latency_ms: number;
}

export interface HealthResponse {
  status: string;
  model: string;
  version: string;
  device: string;
  emotions: string[];
  uptime_seconds: number;
}

export interface AgentAnalysis {
  context: string;
  dominant_emotion: string;
  stress_index: number;
  engagement_score: number;
  interpretation?: string;
  recommendation?: string;
  agent_reasoning?: string;
  llm_augmented: boolean;
}

export interface AgentResponse {
  success: boolean;
  analysis: AgentAnalysis;
}

@Injectable({
  providedIn: 'root'
})
export class FerApiService {
  private http = inject(HttpClient);
  private baseUrl = 'http://localhost:8001';

  getHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/healthz`);
  }

  getInfo(): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/v1/info`);
  }

  predictImage(blob: Blob): Observable<PredictResponse> {
    const formData = new FormData();
    formData.append('file', blob, 'frame.jpg');
    return this.http.post<PredictResponse>(`${this.baseUrl}/v1/predict/image`, formData);
  }

  predictBase64(base64: string): Observable<PredictResponse> {
    return this.http.post<PredictResponse>(`${this.baseUrl}/v1/predict/base64`, {
      image_base64: base64,
      confidence_threshold: 0.35
    });
  }

  analyzeAffectiveBehavior(faceData: FaceDetection, context: string = 'general'): Observable<AgentResponse> {
    return this.http.post<AgentResponse>(`${this.baseUrl}/v1/agent/analyze`, {
      face_data: faceData,
      context: context
    });
  }
}
