import { Component, OnInit, OnDestroy, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { ForexSignalsService } from '../services/forex-signals.service';
import { ForexTensorOpsRequest } from '../../../core/models/forex.models';
import { JobPollingService, JobStatusResponse } from '../../../core/services/job-polling.service';

@Component({
  selector: 'qe-forex-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="forex-dashboard">
      <h2>EUR/USD Forex Signals</h2>

      <div class="controls">
        <label>Volatility Window
          <input type="number" [(ngModel)]="request.volatility_window" min="2" />
        </label>
        <label>Momentum Window
          <input type="number" [(ngModel)]="request.momentum_window" min="2" />
        </label>
        <button (click)="loadSignals()" [disabled]="jobState?.status === 'pending' || jobState?.status === 'running'">
          {{ isPolling ? '🔄 Running…' : 'Run Signals' }}
        </button>
      </div>

      <div *ngIf="error" class="error">{{ error }}</div>

      <!-- Job status badge -->
      <div *ngIf="jobState" class="job-status" [attr.data-status]="jobState.status">
        <strong>Job {{ jobState.job_id }}</strong>:
        {{ statusEmoji(jobState.status) }} {{ jobState.status | uppercase }}
        <span *ngIf="isPolling"> — polling every 3s…</span>
      </div>

      <!-- Results once complete -->
      <div *ngIf="jobState?.status === 'success' && jobState?.result" class="results">
        <h3>Results</h3>
        <table>
          <tr><th>Volatility Points</th><td>{{ jobState!.result!['volatility_points'] | number }}</td></tr>
          <tr><th>Momentum Points</th> <td>{{ jobState!.result!['momentum_points'] | number }}</td></tr>
          <tr><th>NaN Injected</th>    <td>{{ jobState!.result!['nan_injected'] }}</td></tr>
          <tr><th>NaN Remaining</th>   <td>{{ jobState!.result!['nan_remaining'] }}</td></tr>
        </table>
      </div>
    </div>
  `,
})
export class ForexDashboardComponent implements OnInit, OnDestroy {
  private readonly service = inject(ForexSignalsService);
  private readonly polling = inject(JobPollingService);
  private readonly subs    = new Subscription();

  request: ForexTensorOpsRequest = {
    execution_id: crypto.randomUUID(),
    volatility_window: 20,
    momentum_window: 14,
    inject_nan_fraction: 0.01,
  };

  jobState: JobStatusResponse | null = null;
  isPolling = false;
  error: string | null = null;

  ngOnInit(): void {}

  loadSignals(): void {
    this.error = null;
    this.request = { ...this.request, execution_id: crypto.randomUUID() };
    this.subs.add(
      this.service.getSignals(this.request).subscribe({
        next: (submitted) => {
          this.isPolling = true;
          this.jobState = { job_id: submitted.job_id, task_name: submitted.task_name, status: 'pending', submitted_at: new Date().toISOString() };
          this.subs.add(
            this.polling.pollUntilDone(submitted.job_id).subscribe({
              next:     job => { this.jobState = job; },
              complete: ()  => { this.isPolling = false; },
              error:    err => { this.error = err.message; this.isPolling = false; },
            })
          );
        },
        error: (err) => { this.error = err.message; },
      })
    );
  }

  statusEmoji(status: string): string {
    return ({ pending: '⏳', running: '🔄', success: '✅', failed: '❌' } as Record<string, string>)[status] ?? '❓';
  }

  ngOnDestroy(): void { this.subs.unsubscribe(); }
}
