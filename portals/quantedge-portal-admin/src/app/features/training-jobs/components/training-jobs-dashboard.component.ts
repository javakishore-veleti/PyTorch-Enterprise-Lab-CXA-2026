import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  QuantEdgeAdminApiService,
  ForexIngestionResponse, ForexPreprocessResponse,
  ForexAutogradResponse,
} from '../../../core/services/quantedge-admin-api.service';

/** TrainingJobsDashboardComponent — orchestrates the full Forex pipeline from the UI. */
@Component({
  selector: 'qe-admin-training-jobs',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="training-jobs">
      <h2>Training Jobs — Foundations</h2>

      <section class="pipeline">
        <h3>🟡 Forex EUR/USD Pipeline (Week 1)</h3>
        <div class="step" [class.done]="ingestResp?.status === 'success'">
          <span>1. Ingest</span>
          <button (click)="runIngest()" [disabled]="loading">Run</button>
          <code *ngIf="ingestResp">{{ ingestResp.rows_loaded | number }} rows · {{ ingestResp.status }}</code>
        </div>
        <div class="step" [class.done]="preprocessResp?.status === 'success'">
          <span>2. Preprocess</span>
          <button (click)="runPreprocess()" [disabled]="loading || !ingestResp">Run</button>
          <code *ngIf="preprocessResp">NaN filled: {{ preprocessResp.nan_filled }} · {{ preprocessResp.status }}</code>
        </div>
        <div class="step" [class.done]="autogradResp?.status === 'success'">
          <span>3. Autograd Comparison</span>
          <button (click)="runAutograd()" [disabled]="loading || !preprocessResp">Run</button>
          <code *ngIf="autogradResp">
            manual_loss={{ autogradResp.manual_loss | number:'1.6-8' }}
            | autograd_loss={{ autogradResp.autograd_loss | number:'1.6-8' }}
            | grad_diff={{ autogradResp.max_grad_diff }}
          </code>
        </div>
      </section>

      <div *ngIf="error" class="error">{{ error }}</div>
    </div>
  `,
})
export class TrainingJobsDashboardComponent {
  private readonly api = inject(QuantEdgeAdminApiService);

  ingestResp: ForexIngestionResponse | null = null;
  preprocessResp: ForexPreprocessResponse | null = null;
  autogradResp: ForexAutogradResponse | null = null;
  loading = false;
  error: string | null = null;
  private executionId = crypto.randomUUID();

  runIngest(): void {
    this.loading = true;
    this.api.forexIngest({ data_dir: 'data/forex', years: [2019, 2020] }).subscribe({
      next: r => { this.ingestResp = r; this.executionId = r.execution_id; this.loading = false; },
      error: e => { this.error = e.message; this.loading = false; },
    });
  }

  runPreprocess(): void {
    this.loading = true;
    this.api.forexPreprocess({ execution_id: this.executionId }).subscribe({
      next: r => { this.preprocessResp = r; this.loading = false; },
      error: e => { this.error = e.message; this.loading = false; },
    });
  }

  runAutograd(): void {
    this.loading = true;
    this.api.forexAutograd({ execution_id: this.executionId, window_size: 20 }).subscribe({
      next: r => { this.autogradResp = r; this.loading = false; },
      error: e => { this.error = e.message; this.loading = false; },
    });
  }
}
