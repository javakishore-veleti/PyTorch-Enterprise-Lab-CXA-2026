import { Component, inject, signal, computed } from '@angular/core';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { JsonPipe } from '@angular/common';
import { QuantEdgeAdminApiService } from '../../../core/services/quantedge-admin-api.service';
import { JobPollingService } from '../../../core/services/job-polling.service';

@Component({
  selector: 'qe-attention-dashboard',
  standalone: true,
  imports: [ReactiveFormsModule, JsonPipe],
  template: `
    <div style="max-width:1100px">
      <div class="page-header">
        <h1>👁 Attention Transformer</h1>
        <p>Custom multi-head attention from scratch with heatmap visualization · Weeks 5-6</p>
      </div>
    <div class="qe-panel">
      <h2>👁 Attention / Transformer (Week 5-6)</h2>
      <form [formGroup]="form">
        <div class="form-group">
          <label>Execution ID</label>
          <input formControlName="execution_id" type="text" />
        </div>
        <div class="form-group">
          <label>Epochs</label>
          <input formControlName="epochs" type="number" />
        </div>
        <div class="form-group">
          <label>Layer Index (for heatmap)</label>
          <input formControlName="layer_index" type="number" />
        </div>
        <div style="margin-top:16px">
          <button type="button" [disabled]="loading()" (click)="submit('train')">Train</button>
          <button type="button" [disabled]="loading()" (click)="submit('extract')">Extract Weights</button>
          <button type="button" [disabled]="loading()" (click)="submit('heatmap')">Generate Heatmap</button>
          <button type="button" [disabled]="loading()" (click)="submit('arch')">Get Arch Decision</button>
        </div>
      </form>
      @if (currentAction()) {
        <p style="color:#a0aec0;font-size:12px;margin:8px 0">Running: {{ currentAction() }}</p>
      }
      @if (jobId()) {
        <div class="job-status">
          <span class="badge" [class]="statusClass()">{{ jobStatus() }}</span>
          <code>Job: {{ jobId() }}</code>
        </div>
      }
      @if (result()) {
        <pre class="result-box">{{ result() | json }}</pre>
      }
      @if (error()) {
        <div class="error-box">{{ error() }}</div>
      }
    </div>
  `,
})
export class AttentionDashboardComponent {
  private readonly api     = inject(QuantEdgeAdminApiService);
  private readonly polling = inject(JobPollingService);
  private readonly fb      = inject(FormBuilder);

  readonly form = this.fb.group({
    execution_id: ['attn-001'],
    epochs:       [3],
    layer_index:  [0],
  });

  loading       = signal(false);
  jobId         = signal('');
  jobStatus     = signal('');
  result        = signal<any>(null);
  error         = signal('');
  currentAction = signal('');

  statusClass = computed(() => {
    const s = this.jobStatus();
    return {
      'badge-completed': s === 'completed' || s === 'success',
      'badge-failed':    s === 'failed',
      'badge-running':   s === 'running' || s === 'pending',
      'badge-pending':   !s,
    };
  });

  submit(action: 'train' | 'extract' | 'heatmap' | 'arch'): void {
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);
    const v = this.form.value;
    const labels = { train: 'Train', extract: 'Extract Weights', heatmap: 'Generate Heatmap', arch: 'Arch Decision' };
    this.currentAction.set(labels[action]);

    const call$ = action === 'train'   ? this.api.attentionTrain({ execution_id: v.execution_id!, epochs: v.epochs! })
                : action === 'extract' ? this.api.attentionExtract({ execution_id: v.execution_id! })
                : action === 'heatmap' ? this.api.attentionHeatmap({ execution_id: v.execution_id!, layer_index: v.layer_index! })
                : this.api.archDecision({ task_type: 'time_series' });

    call$.subscribe({
      next: (submitted) => {
        this.jobId.set(submitted.job_id);
        this.jobStatus.set('pending');
        this.polling.pollUntilDone(submitted.job_id).subscribe({
          next:     (status) => { this.jobStatus.set(status.status); if (status.result) this.result.set(status.result); },
          error:    (err)    => { this.error.set(err.message); this.loading.set(false); },
          complete: ()       => this.loading.set(false),
        });
      },
      error: (err) => { this.error.set(err.message); this.loading.set(false); },
    });
  }
}
