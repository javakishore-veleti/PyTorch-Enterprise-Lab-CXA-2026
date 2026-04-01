import { Component, inject, signal, computed } from '@angular/core';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { JsonPipe } from '@angular/common';
import { QuantEdgeAdminApiService } from '../../../core/services/quantedge-admin-api.service';
import { JobPollingService } from '../../../core/services/job-polling.service';

@Component({
  selector: 'qe-lora-dashboard',
  standalone: true,
  imports: [ReactiveFormsModule, JsonPipe],
  template: `
    <div style="max-width:1100px">
      <div class="page-header">
        <h1>🔧 LoRA Fine-Tuning</h1>
        <p>Parameter-efficient fine-tuning with Low-Rank Adaptation · Week 7</p>
      </div>
    <div class="qe-panel">
      <h2>🔧 LoRA Fine-Tuning (Week 7)</h2>
      <form [formGroup]="form">
        <div class="form-group">
          <label>Execution ID</label>
          <input formControlName="execution_id" type="text" />
        </div>
        <div class="form-group">
          <label>LoRA Rank</label>
          <input formControlName="lora_rank" type="number" />
        </div>
        <div class="form-group">
          <label>LoRA Alpha</label>
          <input formControlName="lora_alpha" type="number" />
        </div>
        <div class="form-group">
          <label>Epochs</label>
          <input formControlName="epochs" type="number" />
        </div>
        <div style="margin-top:16px">
          <button type="button" [disabled]="loading()" (click)="submit('train')">Train</button>
          <button type="button" [disabled]="loading()" (click)="submit('eval')">Evaluate</button>
          <button type="button" [disabled]="loading()" (click)="submit('predict')">Predict</button>
          <button type="button" [disabled]="loading()" (click)="submit('merge')">Merge Weights</button>
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
export class LoraDashboardComponent {
  private readonly api     = inject(QuantEdgeAdminApiService);
  private readonly polling = inject(JobPollingService);
  private readonly fb      = inject(FormBuilder);

  readonly form = this.fb.group({
    execution_id: ['lora-001'],
    lora_rank:    [16],
    lora_alpha:   [32],
    epochs:       [5],
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

  submit(action: 'train' | 'eval' | 'predict' | 'merge'): void {
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);
    const v = this.form.value;
    const req = { execution_id: v.execution_id!, lora_rank: v.lora_rank!, lora_alpha: v.lora_alpha!, epochs: v.epochs! };
    const labels = { train: 'Train', eval: 'Evaluate', predict: 'Predict', merge: 'Merge Weights' };
    this.currentAction.set(labels[action]);

    const call$ = action === 'train'   ? this.api.loraTrain(req)
                : action === 'eval'    ? this.api.loraEval(req)
                : action === 'predict' ? this.api.loraPredict(req)
                : this.api.loraMerge(req);

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
