import { Component, inject, signal, computed } from '@angular/core';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { JsonPipe } from '@angular/common';
import { QuantEdgeAdminApiService } from '../../../core/services/quantedge-admin-api.service';
import { JobPollingService } from '../../../core/services/job-polling.service';

@Component({
  selector: 'qe-drift-monitoring-dashboard',
  standalone: true,
  imports: [ReactiveFormsModule, JsonPipe],
  template: `
    <div class="qe-panel">
      <h2>📡 Drift & Monitoring (Week 12)</h2>
      <form [formGroup]="form">
        <div class="form-group">
          <label>PSI Threshold (data drift)</label>
          <input formControlName="psi_threshold" type="number" step="0.01" />
        </div>
        <div class="form-group">
          <label>Window Size (concept drift)</label>
          <input formControlName="window_size" type="number" />
        </div>
        <div class="form-group">
          <label>Audit Event Type</label>
          <input formControlName="event_type" type="text" />
        </div>
        <div class="form-group">
          <label>Actor</label>
          <input formControlName="actor" type="text" />
        </div>
        <div class="form-group">
          <label>Resource</label>
          <input formControlName="resource" type="text" />
        </div>
        <div style="margin-top:16px">
          <button type="button" [disabled]="loading()" (click)="submit('data-drift')">Detect Data Drift</button>
          <button type="button" [disabled]="loading()" (click)="submit('concept-drift')">Detect Concept Drift</button>
          <button type="button" [disabled]="loading()" (click)="submit('prometheus')">Get Prometheus Metrics</button>
          <button type="button" [disabled]="loading()" (click)="submit('audit')">Log Audit Event</button>
          <button type="button" [disabled]="loading()" (click)="submit('adr-gen')">Generate ADRs</button>
          <button type="button" [disabled]="loading()" (click)="submit('adr-list')">List ADRs</button>
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
export class DriftMonitoringDashboardComponent {
  private readonly api     = inject(QuantEdgeAdminApiService);
  private readonly polling = inject(JobPollingService);
  private readonly fb      = inject(FormBuilder);

  readonly form = this.fb.group({
    psi_threshold: [0.2],
    window_size:   [100],
    event_type:    ['model_deployed'],
    actor:         ['admin'],
    resource:      ['ForexTransformer'],
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

  submit(action: 'data-drift' | 'concept-drift' | 'prometheus' | 'audit' | 'adr-gen' | 'adr-list'): void {
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);
    const v = this.form.value;
    const labels: Record<string, string> = {
      'data-drift': 'Detect Data Drift', 'concept-drift': 'Detect Concept Drift',
      prometheus: 'Prometheus Metrics', audit: 'Log Audit Event',
      'adr-gen': 'Generate ADRs', 'adr-list': 'List ADRs',
    };
    this.currentAction.set(labels[action]);

    const call$ = action === 'data-drift'    ? this.api.dataDrift({ psi_threshold: v.psi_threshold! })
                : action === 'concept-drift' ? this.api.conceptDrift({ window_size: v.window_size! })
                : action === 'prometheus'    ? this.api.prometheusMetrics({})
                : action === 'audit'         ? this.api.auditLog({ event_type: v.event_type!, actor: v.actor!, resource: v.resource! })
                : action === 'adr-gen'       ? this.api.adrGenerate({})
                : this.api.adrList({});

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
