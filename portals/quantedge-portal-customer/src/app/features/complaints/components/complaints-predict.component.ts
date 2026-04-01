import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ComplaintsAnalyticsService } from '../services/complaints-analytics.service';
import { CFPBPredictRequest, CFPBPredictResponse } from '../../../core/models/cfpb.models';

@Component({
  selector: 'qe-complaints-predict',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="complaints-predict">
      <h2>Banking Complaint Classifier</h2>
      <p class="subtitle">Powered by CFPB Consumer Financial Complaints · 3M+ records</p>

      <div class="controls">
        <label>Complaint Narrative
          <textarea [(ngModel)]="complaintText" rows="5" placeholder="Describe the complaint…"></textarea>
        </label>
        <label>Checkpoint Path
          <input [(ngModel)]="checkpointPath" placeholder="data/checkpoints/cfpb/epoch_002.pt" />
        </label>
        <button (click)="classify()" [disabled]="loading || !complaintText">
          {{ loading ? 'Classifying…' : 'Classify Product' }}
        </button>
      </div>

      <div *ngIf="error" class="error">{{ error }}</div>

      <div *ngIf="response" class="results">
        <h3>Prediction <span [class]="response.status">{{ response.status }}</span></h3>
        <p><strong>Product Category:</strong> {{ response.predicted_product }}</p>
        <p><strong>Confidence:</strong> {{ response.confidence | percent:'1.1-2' }}</p>
      </div>
    </div>
  `,
})
export class ComplaintsPredictComponent {
  private readonly service = inject(ComplaintsAnalyticsService);

  complaintText = '';
  checkpointPath = 'data/checkpoints/cfpb/epoch_002.pt';
  response: CFPBPredictResponse | null = null;
  loading = false;
  error: string | null = null;

  classify(): void {
    this.loading = true;
    this.error = null;
    const request: CFPBPredictRequest = {
      execution_id: crypto.randomUUID(),
      text: this.complaintText,
      checkpoint_path: this.checkpointPath,
    };
    this.service.predict(request).subscribe({
      next: (resp) => { this.response = resp; this.loading = false; },
      error: (err) => { this.error = err.message; this.loading = false; },
    });
  }
}
