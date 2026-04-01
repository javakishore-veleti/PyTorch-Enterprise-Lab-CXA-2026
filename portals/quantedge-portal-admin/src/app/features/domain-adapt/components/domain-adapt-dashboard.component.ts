import { Component, inject, signal, computed } from '@angular/core';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { JsonPipe } from '@angular/common';
import { QuantEdgeAdminApiService } from '../../../core/services/quantedge-admin-api.service';
import { JobPollingService } from '../../../core/services/job-polling.service';

@Component({
  selector: 'qe-domain-adapt-dashboard',
  standalone: true,
  imports: [ReactiveFormsModule, JsonPipe],
  template: `
    <div style="max-width:1100px">
      <div class="page-header">
        <h1>🤖 Domain Adaptation &amp; Ollama</h1>
        <p>PEFT domain adaptation + local Ollama LLM serving · Week 8</p>
      </div>
    <div class="qe-panel">
      <h2>🤖 Domain Adapt + Ollama (Week 8)</h2>
      <form [formGroup]="form">
        <div class="form-group">
          <label>Model Name</label>
          <input formControlName="model_name" type="text" />
        </div>
        <div class="form-group">
          <label>Max Steps</label>
          <input formControlName="max_steps" type="number" />
        </div>
        <div class="form-group">
          <label>Prompt (for Ollama)</label>
          <textarea formControlName="prompt" rows="3"></textarea>
        </div>
        <div style="margin-top:16px">
          <button type="button" [disabled]="loading()" (click)="submit('adapt-train')">Domain Adapt Train</button>
          <button type="button" [disabled]="loading()" (click)="submit('adapt-eval')">Domain Adapt Eval</button>
          <button type="button" [disabled]="loading()" (click)="submit('ollama-infer')">Ollama Infer</button>
          <button type="button" [disabled]="loading()" (click)="submit('ollama-merge')">Ollama Merge</button>
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
export class DomainAdaptDashboardComponent {
  private readonly api     = inject(QuantEdgeAdminApiService);
  private readonly polling = inject(JobPollingService);
  private readonly fb      = inject(FormBuilder);

  readonly form = this.fb.group({
    model_name: ['gpt2'],
    max_steps:  [10],
    prompt:     ['Summarize financial risks for Q1 2024'],
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

  submit(action: 'adapt-train' | 'adapt-eval' | 'ollama-infer' | 'ollama-merge'): void {
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);
    const v = this.form.value;
    const labels: Record<string, string> = { 'adapt-train': 'Domain Adapt Train', 'adapt-eval': 'Domain Adapt Eval', 'ollama-infer': 'Ollama Infer', 'ollama-merge': 'Ollama Merge' };
    this.currentAction.set(labels[action]);

    const call$ = action === 'adapt-train'   ? this.api.domainAdaptTrain({ model_name: v.model_name!, max_steps: v.max_steps! })
                : action === 'adapt-eval'    ? this.api.domainAdaptEval({ model_name: v.model_name! })
                : action === 'ollama-infer'  ? this.api.ollamaInfer({ prompt: v.prompt!, model_name: v.model_name! })
                : this.api.ollamaMerge({ base_model_name: v.model_name! });

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
