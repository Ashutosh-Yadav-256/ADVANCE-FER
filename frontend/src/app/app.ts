import { Component, OnInit, OnDestroy, ViewChild, ElementRef, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { FerApiService, FaceDetection, HealthResponse, AgentAnalysis } from './services/fer-api.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit, OnDestroy {
  private api = inject(FerApiService);

  @ViewChild('videoPlayer') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('hudCanvas') canvasElement!: ElementRef<HTMLCanvasElement>;
  @ViewChild('uploadedCanvas') uploadedCanvas!: ElementRef<HTMLCanvasElement>;

  // Navigation State
  activeTab = signal<'studio' | 'medialab' | 'copilot' | 'metrics'>('studio');

  // Webcam & Pipeline State
  isCameraActive = signal(false);
  isAnalyzing = signal(false);
  backendHealthy = signal(false);
  fps = signal(0);
  latency = signal(0);

  // Settings
  confidenceThreshold = 0.35;
  drawLandmarks = false;
  drawBbox = true;
  selectedContext = 'interview';

  // Detection Results
  faces = signal<FaceDetection[]>([]);
  activeFace = signal<FaceDetection | null>(null);
  health = signal<HealthResponse | null>(null);
  agentAnalysis = signal<AgentAnalysis | null>(null);

  // Media Lab State
  uploadedImage: string | null = null;
  uploadedFaces = signal<FaceDetection[]>([]);
  isUploading = signal(false);

  private videoStream: MediaStream | null = null;
  private animationFrameId: number | null = null;
  private lastFrameTime = performance.now();
  private frameCount = 0;
  private fpsInterval: any;

  // Apple System Palette for Emotions
  emotionColors: Record<string, string> = {
    happy: '#34C759',    // SF Green
    surprise: '#0071E3', // SF Blue
    neutral: '#8E8E93',  // SF Gray
    sad: '#5856D6',      // SF Indigo
    fear: '#FF9500',     // SF Orange
    angry: '#FF3B30',    // SF Red
    disgust: '#30B0C7'   // SF Teal
  };

  ngOnInit() {
    this.checkHealth();
    // Poll health every 10 seconds
    setInterval(() => this.checkHealth(), 10000);
  }

  ngOnDestroy() {
    this.stopCamera();
    if (this.fpsInterval) clearInterval(this.fpsInterval);
  }

  checkHealth() {
    this.api.getHealth().subscribe({
      next: (res) => {
        this.backendHealthy.set(true);
        this.health.set(res);
      },
      error: () => {
        this.backendHealthy.set(false);
      }
    });
  }

  setTab(tab: 'studio' | 'medialab' | 'copilot' | 'metrics') {
    this.activeTab.set(tab);
    if (tab !== 'studio' && this.isCameraActive()) {
      this.stopCamera();
    }
  }

  // ==========================================
  // WEBCAM STUDIO
  // ==========================================

  async startCamera() {
    try {
      this.videoStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false
      });

      if (this.videoElement && this.videoElement.nativeElement) {
        this.videoElement.nativeElement.srcObject = this.videoStream;
        this.videoElement.nativeElement.play();
        this.isCameraActive.set(true);

        this.fpsInterval = setInterval(() => {
          this.fps.set(this.frameCount);
          this.frameCount = 0;
        }, 1000);

        this.processWebcamLoop();
      }
    } catch (err) {
      console.error('Camera access error:', err);
      alert('Could not access webcam. Please verify camera permissions.');
    }
  }

  stopCamera() {
    if (this.videoStream) {
      this.videoStream.getTracks().forEach(t => t.stop());
      this.videoStream = null;
    }
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    if (this.fpsInterval) clearInterval(this.fpsInterval);
    this.isCameraActive.set(false);
    this.faces.set([]);
    this.activeFace.set(null);
  }

  private processWebcamLoop = async () => {
    if (!this.isCameraActive() || !this.videoElement) return;

    const video = this.videoElement.nativeElement;
    const canvas = this.canvasElement?.nativeElement;

    if (video.readyState >= 2 && canvas && !this.isAnalyzing()) {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Capture frame to blob
        const offscreen = document.createElement('canvas');
        offscreen.width = canvas.width;
        offscreen.height = canvas.height;
        const offCtx = offscreen.getContext('2d');
        if (offCtx) {
          offCtx.drawImage(video, 0, 0);
          offscreen.toBlob(async (blob) => {
            if (blob && !this.isAnalyzing()) {
              this.isAnalyzing.set(true);
              const t0 = performance.now();
              this.api.predictImage(blob).subscribe({
                next: (res) => {
                  this.latency.set(Math.round(performance.now() - t0));
                  this.frameCount++;
                  this.faces.set(res.faces);
                  if (res.faces.length > 0) {
                    this.activeFace.set(res.faces[0]);
                  }
                  this.drawHUD(ctx, res.faces, canvas.width, canvas.height);
                  this.isAnalyzing.set(false);
                },
                error: () => {
                  this.isAnalyzing.set(false);
                }
              });
            }
          }, 'image/jpeg', 0.85);
        }
      }
    }

    // Schedule next frame loop with slight pacing
    setTimeout(() => {
      if (this.isCameraActive()) {
        this.animationFrameId = requestAnimationFrame(this.processWebcamLoop);
      }
    }, 60);
  };

  private drawHUD(ctx: CanvasRenderingContext2D, faces: FaceDetection[], w: number, h: number) {
    if (!this.drawBbox) return;

    faces.forEach((face, idx) => {
      const [x, y, bw, bh] = face.bbox;
      const emotion = face.dominant_emotion.toLowerCase();
      const color = this.emotionColors[emotion] || '#10B981';

      // Futuristic bounding box corners
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.strokeRect(x, y, bw, bh);
      ctx.shadowBlur = 0; // reset

      // Header Tag
      const label = `${face.dominant_emotion.toUpperCase()} ${(face.confidence * 100).toFixed(0)}%`;
      ctx.font = '600 13px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
      const textWidth = ctx.measureText(label).width;

      const badgeY = Math.max(22, y);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x, badgeY - 20, textWidth + 14, 20, [4, 4, 0, 0]);
      ctx.fill();

      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(label, x + 7, badgeY - 5);

      // Valence/Arousal sub-tag
      const vaText = `V:${face.valence > 0 ? '+' : ''}${face.valence.toFixed(2)} A:${face.arousal > 0 ? '+' : ''}${face.arousal.toFixed(2)}`;
      ctx.font = '500 11px "SF Mono", "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.88)';
      ctx.fillRect(x, y + bh, 125, 18);
      ctx.fillStyle = '#1d1d1f';
      ctx.fillText(vaText, x + 5, y + bh + 13);
    });
  }

  // ==========================================
  // AFFECTIVE AGENT REASONING
  // ==========================================

  triggerAgentAnalysis() {
    const face = this.activeFace();
    if (!face) {
      alert('No face detected to analyze. Please ensure a face is visible.');
      return;
    }

    this.api.analyzeAffectiveBehavior(face, this.selectedContext).subscribe({
      next: (res) => {
        this.agentAnalysis.set(res.analysis);
      },
      error: (err) => {
        console.error('Agent analysis error:', err);
      }
    });
  }

  runRealInferenceOnSample(imagePath: string = '/sample_face.jpg') {
    this.uploadedImage = imagePath;
    this.isUploading.set(true);

    fetch(imagePath)
      .then(res => res.blob())
      .then(blob => {
        this.api.predictImage(blob).subscribe({
          next: (res) => {
            this.uploadedFaces.set(res.faces);
            if (res.faces.length > 0) {
              this.activeFace.set(res.faces[0]);
              this.triggerAgentAnalysis();
            }
            this.isUploading.set(false);
          },
          error: (err) => {
            console.error('Real model inference failed on backend:', err);
            this.isUploading.set(false);
          }
        });
      })
      .catch(err => {
        console.error('Error fetching sample image:', err);
        this.isUploading.set(false);
      });
  }

  // ==========================================
  // MEDIA LAB (FILE UPLOAD)
  // ==========================================

  onFileSelected(event: any) {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e: any) => {
      this.uploadedImage = e.target.result;
      this.isUploading.set(true);

      this.api.predictImage(file).subscribe({
        next: (res) => {
          this.uploadedFaces.set(res.faces);
          if (res.faces.length > 0) {
            this.activeFace.set(res.faces[0]);
          }
          this.isUploading.set(false);
        },
        error: (err) => {
          console.error('Upload prediction failed:', err);
          this.isUploading.set(false);
        }
      });
    };
    reader.readAsDataURL(file);
  }

  getSortedProbabilities(face: FaceDetection) {
    if (!face || !face.probabilities) return [];
    return Object.entries(face.probabilities)
      .map(([emotion, prob]) => ({
        emotion,
        prob: Math.round(prob * 100),
        color: this.emotionColors[emotion.toLowerCase()] || '#94A3B8'
      }))
      .sort((a, b) => b.prob - a.prob);
  }
}
