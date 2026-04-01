import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ForexSignalsService } from '../services/forex-signals.service';
import { ForexTensorOpsRequest, ForexTensorOpsResponse } from '../../../core/models/forex.models';

@Component({
  selector: 'qe-forex-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="forex-dashboard">
      <h2>EUR/USD Forex Signals</h2>

      <div class="controls">
        <label>Execution ID
          <input [(ngModel)]="request.execution_id" placeholder="uuid" />
        </label>
        <label>Volatility Window
          <input type="number" [(ngModel)]="request.volatility_window" min="2" />
        </label>
        <label>Momentum Window
          <input type="number" [(ngModel)]="request.momentum_window" min="2" />
        </label>
        <button (click)="loadSignals()" [disabled]="loading">
          {{ loading ? 'Loading…' : 'Run Signals' }}
        </button>
      </div>

      <div *ngIf="error" class="error">{{ error }}</div>

      <div *ngIf="response" class="results">
        <h3>Results <span [class]="response.status">{{ response.status }}</span></h3>
        <table>
          <tr><th>Volatility Points</th><td>{{ response.volatility_points | number }}</td></tr>
          <tr><th>Momentum Points</th><td>{{ response.momentum_points | number }}</td></tr>
          <tr><th>NaN Injected</th><td>{{ response.nan_injected }}</td></tr>
          <tr><th>NaN Remaining</th><td>{{ response.nan_remaining }}</td></tr>
        </table>
      </div>
    </div>
  `,
})
export class ForexDashboardComponent implements OnInit {
  private readonly service = inject(ForexSignalsService);

  request: ForexTensorOpsRequest = {
    execution_id: crypto.randomUUID(),
    volatility_window: 20,
    momentum_window: 14,
    inject_nan_fraction: 0.01,
  };

  response: ForexTensorOpsResponse | null = null;
  loading = false;
  error: string | null = null;

  ngOnInit(): void {}

  loadSignals(): void {
    this.loading = true;
    this.error = null;
    this.service.getSignals(this.request).subscribe({
      next: (resp) => { this.response = resp; this.loading = false; },
      error: (err) => { this.error = err.message; this.loading = false; },
    });
  }
}
