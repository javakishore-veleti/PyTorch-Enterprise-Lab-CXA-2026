import { Component, inject, signal, computed } from '@angular/core';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { JsonPipe } from '@angular/common';
import { QuantEdgeAdminApiService } from '../../../core/services/quantedge-admin-api.service';
import { JobPollingService } from '../../../core/services/job-polling.service';

@Component({
  selector: 'qe-quantization-dashboard',
  standalone: true,
  imports: [ReactiveFormsModule, JsonPipe],
  template: `
    <div class="qe-panel">
      <h2>🗜 Quantization + Serving (Week 10)</h2>
      <form [formGroup]="form">
        <div class="form-group">
          <label>Output Directory</label>
          <input formControlName="output_dir" type="text" />
        </div>
        <div class="form-group">
          <label>Calibration Batches</label>
          <input formControlName="calibration_batches" type="number" />
        </div>
        <div class="form-group">
          <label>QAT Train Steps</label>
          <input formControlName="train_steps" type="number" />
        </div>
        <div class="form-group">
          <label>Model Format (for inference/serving)</label>
          <select formControlName="model_format">
            <option value="eager">eager</option>
            <option value="torchscript">torchscript</option>
            <option value="quantized_dynamic">quantized_dynamic</option>
          </select>
        </div>
        <div style="margin-top:16px">
          <button type="button" [disabled]="loading()" (click)="submit('static')">Quantize Static</button>
          <button type="button" [disabled]="loading()" (click)="submit('dynamic')">Quantize Dynamic</button>
          <button type="button" [disabled]="loading()" (click)="submit('qat')">Quantize QAT</button>
          <button type="button" [disabled]="loading()" (click)="submit('compare')">Compare</button>
          <button type="button" [disabled]="loading()" (click)="submit('infer')">Model Infer</button>
          <button type="button" [disabled]="loading()" (click)="submit('bench')">Serving Benchmark</button>
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
export class QuantizationDashboardComponent {
  private readonly api     = inject(QuantEdgeAdminApiService);
  private readonly polling = inject(JobPollingService);
  private readonly fb      = inject(FormBuilder);

  readonly form = this.fb.group({
    output_dir:          ['data/quantized'],
    calibration_batches: [5],
    train_steps:         [10],
    model_format:        ['eager'],
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

  submit(action: 'static' | 'dynamic' | 'qat' | 'compare' | 'infer' | 'bench'): void {
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);
    const v = this.form.value;
    const labels: Record<string, string> = {
      static: 'Quantize Static', dynamic: 'Quantize Dynamic', qat: 'Quantize QAT',
      compare: 'Compare', infer: 'Model Infer', bench: 'Serving Benchmark',
    };
    this.currentAction.set(labels[action]);

    const call$ = action === 'static'  ? this.api.quantizeStatic({ output_dir: v.output_dir!, calibration_batches: v.calibration_batches! })
                : action === 'dynamic' ? this.api.quantizeDynamic({ output_dir: v.output_dir! })
                : action === 'qat'     ? this.api.quantizeQat({ output_dir: v.output_dir!, train_steps: v.train_steps! })
                : action === 'compare' ? this.api.quantizeCompare({ output_dir: v.output_dir! })
                : action === 'infer'   ? this.api.modelInfer({ model_format: v.model_format! })
                : this.api.servingBenchmark({ model_format: v.model_format! });

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
