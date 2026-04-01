import { Component, inject, signal, computed } from '@angular/core';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { JsonPipe } from '@angular/common';
import { QuantEdgeAdminApiService } from '../../../core/services/quantedge-admin-api.service';
import { JobPollingService } from '../../../core/services/job-polling.service';

@Component({
  selector: 'qe-export-dashboard',
  standalone: true,
  imports: [ReactiveFormsModule, JsonPipe],
  template: `
    <div style="max-width:1100px">
      <div class="page-header">
        <h1>📦 Model Export Formats</h1>
        <p>TorchScript trace · ONNX opset 17 · TensorRT · Inference benchmark · Week 9</p>
      </div>
    <div class="qe-panel">
      <h2>📦 Export Formats (Week 9)</h2>
      <form [formGroup]="form">
        <div class="form-group">
          <label>Output Directory</label>
          <input formControlName="output_dir" type="text" />
        </div>
        <div class="form-group">
          <label>Export Mode</label>
          <select formControlName="export_mode">
            <option value="trace">trace</option>
            <option value="script">script</option>
          </select>
        </div>
        <div class="form-group">
          <label>TensorRT Precision</label>
          <select formControlName="precision">
            <option value="fp32">fp32</option>
            <option value="fp16">fp16</option>
            <option value="int8">int8</option>
          </select>
        </div>
        <div class="form-group">
          <label>Benchmark Runs</label>
          <input formControlName="num_runs" type="number" />
        </div>
        <div style="margin-top:16px">
          <button type="button" [disabled]="loading()" (click)="submit('torchscript')">Export TorchScript</button>
          <button type="button" [disabled]="loading()" (click)="submit('onnx')">Export ONNX</button>
          <button type="button" [disabled]="loading()" (click)="submit('validate-onnx')">Validate ONNX</button>
          <button type="button" [disabled]="loading()" (click)="submit('tensorrt')">Export TensorRT</button>
          <button type="button" [disabled]="loading()" (click)="submit('benchmark')">Benchmark</button>
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
export class ExportDashboardComponent {
  private readonly api     = inject(QuantEdgeAdminApiService);
  private readonly polling = inject(JobPollingService);
  private readonly fb      = inject(FormBuilder);

  readonly form = this.fb.group({
    output_dir:  ['data/exports'],
    export_mode: ['trace'],
    precision:   ['fp16'],
    num_runs:    [50],
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

  submit(action: 'torchscript' | 'onnx' | 'validate-onnx' | 'tensorrt' | 'benchmark'): void {
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);
    const v = this.form.value;
    const labels: Record<string, string> = {
      torchscript: 'Export TorchScript', onnx: 'Export ONNX',
      'validate-onnx': 'Validate ONNX', tensorrt: 'Export TensorRT', benchmark: 'Benchmark',
    };
    this.currentAction.set(labels[action]);

    const call$ = action === 'torchscript'   ? this.api.exportTorchScript({ output_dir: v.output_dir!, export_mode: v.export_mode! })
                : action === 'onnx'          ? this.api.exportOnnx({ output_dir: v.output_dir! })
                : action === 'validate-onnx' ? this.api.validateOnnx({ onnx_path: `${v.output_dir}/model.onnx` })
                : action === 'tensorrt'      ? this.api.exportTensorRT({ torchscript_path: `${v.output_dir}/model.pt`, output_dir: v.output_dir!, precision: v.precision! })
                : this.api.benchmark({ num_runs: v.num_runs! });

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
