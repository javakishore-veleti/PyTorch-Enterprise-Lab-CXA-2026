import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  QuantEdgeAdminApiService,
  CFPBIngestionResponse, CFPBPreprocessResponse,
  CFPBDatasetResponse, CFPBTrainResponse,
} from '../../../core/services/quantedge-admin-api.service';

@Component({
  selector: 'qe-admin-banking-data',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="banking-data">
      <h2>CFPB Complaints Pipeline (Week 2)</h2>

      <div class="step" [class.done]="ingestResp?.status === 'success'">
        <span>1. Ingest from HuggingFace</span>
        <button (click)="runIngest()" [disabled]="loading">Run</button>
        <code *ngIf="ingestResp">{{ ingestResp.rows_loaded | number }} rows · {{ ingestResp.status }}</code>
      </div>
      <div class="step" [class.done]="preprocessResp?.status === 'success'">
        <span>2. Tokenize + Class Weights</span>
        <button (click)="runPreprocess()" [disabled]="loading || !ingestResp">Run</button>
        <code *ngIf="preprocessResp">{{ preprocessResp.n_classes }} classes · {{ preprocessResp.rows_after_filter | number }} rows</code>
      </div>
      <div class="step" [class.done]="datasetResp?.status === 'success'">
        <span>3. Build DataLoaders</span>
        <button (click)="runDataset()" [disabled]="loading || !preprocessResp">Run</button>
        <code *ngIf="datasetResp">train={{ datasetResp.train_samples | number }} val={{ datasetResp.val_samples | number }}</code>
      </div>
      <div class="step" [class.done]="trainResp?.status === 'success'">
        <span>4. Train Model</span>
        <button (click)="runTrain()" [disabled]="loading || !datasetResp">Run (epochs={{ epochs }})</button>
        <input type="number" [(ngModel)]="epochs" min="1" max="10" style="width:50px" />
        <code *ngIf="trainResp">
          val_acc={{ trainResp.final_val_accuracy | percent }} ·
          val_loss={{ trainResp.final_val_loss | number:'1.4-6' }}
        </code>
      </div>

      <div *ngIf="error" class="error">{{ error }}</div>
    </div>
  `,
})
export class BankingDataDashboardComponent {
  private readonly api = inject(QuantEdgeAdminApiService);

  ingestResp: CFPBIngestionResponse | null = null;
  preprocessResp: CFPBPreprocessResponse | null = null;
  datasetResp: CFPBDatasetResponse | null = null;
  trainResp: CFPBTrainResponse | null = null;
  loading = false;
  error: string | null = null;
  epochs = 3;
  private executionId = crypto.randomUUID();

  runIngest(): void {
    this.loading = true;
    this.api.cfpbIngest({ cache_dir: 'data/cfpb' }).subscribe({
      next: r => { this.ingestResp = r; this.executionId = r.execution_id; this.loading = false; },
      error: e => { this.error = e.message; this.loading = false; },
    });
  }

  runPreprocess(): void {
    this.loading = true;
    this.api.cfpbPreprocess({ execution_id: this.executionId }).subscribe({
      next: r => { this.preprocessResp = r; this.loading = false; },
      error: e => { this.error = e.message; this.loading = false; },
    });
  }

  runDataset(): void {
    this.loading = true;
    this.api.cfpbBuildDataloaders({ execution_id: this.executionId }).subscribe({
      next: r => { this.datasetResp = r; this.loading = false; },
      error: e => { this.error = e.message; this.loading = false; },
    });
  }

  runTrain(): void {
    this.loading = true;
    this.api.cfpbTrain({ execution_id: this.executionId, epochs: this.epochs }).subscribe({
      next: r => { this.trainResp = r; this.loading = false; },
      error: e => { this.error = e.message; this.loading = false; },
    });
  }
}
