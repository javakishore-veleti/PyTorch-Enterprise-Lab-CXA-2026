import { Component, OnInit, OnDestroy, inject } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { ForexSignalsService } from '../services/forex-signals.service';
import { ForexTensorOpsRequest } from '../../../core/models/forex.models';
import { JobPollingService, JobStatusResponse } from '../../../core/services/job-polling.service';

@Component({
  selector: 'qe-forex-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, DecimalPipe],
  template: `
    <div class="forex-dashboard">
      <h2>💹 EUR/USD Forex Signals</h2>

      <!-- Signal Strength Indicator -->
      @if (jobState?.status === 'success' && jobState?.result) {
        <div class="signal-strength-panel">
          <div class="signal-label">Signal Strength</div>
          <div class="signal-bar-wrap">
            <div class="signal-bar" [style.width]="signalPct() + '%'" [class]="signalClass()"></div>
          </div>
          <div class="signal-value">{{ signalPct() | number:'1.1-1' }}%</div>
        </div>
      }

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

      <!-- Job History -->
      @if (jobHistory.length > 0) {
        <div class="job-history">
          <h3>Recent Forex Jobs</h3>
          <table class="history-table">
            <thead>
              <tr><th>Job ID</th><th>Status</th><th>Time</th></tr>
            </thead>
            <tbody>
              @for (job of jobHistory; track job.job_id) {
                <tr>
                  <td><code>{{ job.job_id.slice(0,8) }}…</code></td>
                  <td>{{ statusEmoji(job.status) }} {{ job.status }}</td>
                  <td>{{ job.submitted_at | date:'HH:mm:ss' }}</td>
                </tr>
              }
            </tbody>
          </table>
        </div>
      }
    </div>
  `,
  styles: [`
    .signal-strength-panel { background: #1a1d27; border: 1px solid #2d3748; border-radius: 8px; padding: 16px; margin-bottom: 20px; display: flex; align-items: center; gap: 16px; }
    .signal-label { color: #a0aec0; font-size: 13px; width: 120px; }
    .signal-bar-wrap { flex: 1; height: 8px; background: #2d3748; border-radius: 4px; overflow: hidden; }
    .signal-bar { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
    .signal-strong { background: #68d391; }
    .signal-medium { background: #f6ad55; }
    .signal-weak { background: #fc8181; }
    .signal-value { color: #e2e8f0; font-size: 14px; font-weight: 600; width: 50px; text-align: right; }
    .job-history { margin-top: 24px; }
    .history-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .history-table th { text-align: left; color: #718096; padding: 6px 12px; border-bottom: 1px solid #2d3748; font-size: 11px; text-transform: uppercase; }
    .history-table td { padding: 8px 12px; border-bottom: 1px solid #1a1d27; color: #e2e8f0; }
  `],
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
  jobHistory: JobStatusResponse[] = [];
  isPolling = false;
  error: string | null = null;

  ngOnInit(): void {}

  signalPct(): number {
    const r = this.jobState?.result;
    if (!r) return 0;
    const vol = r['volatility_points'] ?? 0;
    const mom = r['momentum_points'] ?? 0;
    return Math.min(100, Math.round(((vol + mom) / 2000) * 100));
  }

  signalClass(): string {
    const pct = this.signalPct();
    if (pct >= 65) return 'signal-strong';
    if (pct >= 35) return 'signal-medium';
    return 'signal-weak';
  }

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
              complete: ()  => {
                this.isPolling = false;
                if (this.jobState) this.jobHistory = [this.jobState, ...this.jobHistory].slice(0, 10);
              },
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
