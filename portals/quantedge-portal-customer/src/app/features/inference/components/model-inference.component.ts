import { Component, inject, signal, computed } from '@angular/core';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { JsonPipe } from '@angular/common';
import { QuantEdgeApiService } from '../../../core/services/quantedge-api.service';
import { JobPollingService } from '../../../core/services/job-polling.service';

@Component({
  selector: 'qe-model-inference',
  standalone: true,
  imports: [ReactiveFormsModule, JsonPipe],
  template: `
    <div class="qe-panel">
      <h2>⚡ Model Inference</h2>
      <p style="color:#718096;font-size:13px;margin-bottom:20px">
        Submit inference requests against deployed PyTorch models.
      </p>
      <form [formGroup]="form" (ngSubmit)="submit()">
        <div class="form-group">
          <label>Model Format</label>
          <select formControlName="model_format">
            <option value="eager">Eager (PyTorch)</option>
            <option value="torchscript">TorchScript</option>
            <option value="quantized_dynamic">Quantized Dynamic</option>
          </select>
        </div>
        <div class="form-group">
          <label>Input Size (features)</label>
          <input formControlName="input_size" type="number" />
        </div>
        <div class="form-group">
          <label>Sequence Length</label>
          <input formControlName="seq_len" type="number" />
        </div>
        <button type="submit" [disabled]="loading()">
          {{ loading() ? 'Running…' : 'Run Inference' }}
        </button>
      </form>
      @if (jobId()) {
        <div class="job-status">
          <span class="badge" [class]="statusClass()">{{ jobStatus() }}</span>
          <code>Job: {{ jobId() }}</code>
        </div>
      }
      @if (result()) {
        <div class="qe-panel" style="margin-top:16px">
          <h3 style="color:#68d391;margin:0 0 12px">Inference Result</h3>
          @if (scalarResult() !== null) {
            <div style="font-size:32px;color:#68d391;font-weight:700">{{ scalarResult() | number:'1.4-6' }}</div>
            <div style="color:#718096;font-size:12px;margin-top:4px">Predicted value</div>
          }
          <pre class="result-box">{{ result() | json }}</pre>
        </div>
      }
      @if (error()) {
        <div class="error-box">{{ error() }}</div>
      }
    </div>
  `,
})
export class ModelInferenceComponent {
  private readonly api     = inject(QuantEdgeApiService);
  private readonly polling = inject(JobPollingService);
  private readonly fb      = inject(FormBuilder);

  readonly form = this.fb.group({
    model_format: ['eager'],
    input_size:   [14],
    seq_len:      [30],
  });

  loading   = signal(false);
  jobId     = signal('');
  jobStatus = signal('');
  result    = signal<any>(null);
  error     = signal('');

  scalarResult = computed(() => {
    const r = this.result();
    if (!r) return null;
    if (typeof r === 'number') return r;
    if (typeof r?.prediction === 'number') return r.prediction;
    if (typeof r?.value === 'number') return r.value;
    return null;
  });

  statusClass = computed(() => {
    const s = this.jobStatus();
    return {
      'badge-completed': s === 'completed' || s === 'success',
      'badge-failed':    s === 'failed',
      'badge-running':   s === 'running' || s === 'pending',
      'badge-pending':   !s,
    };
  });

  submit(): void {
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);
    const v = this.form.value;

    this.api.requestInference({ model_format: v.model_format!, input_size: v.input_size!, seq_len: v.seq_len! }).subscribe({
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
