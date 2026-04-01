import { Component, inject, signal, computed } from '@angular/core';
import { ReactiveFormsModule, FormBuilder } from '@angular/forms';
import { JsonPipe } from '@angular/common';
import { QuantEdgeAdminApiService } from '../../../core/services/quantedge-admin-api.service';
import { JobPollingService } from '../../../core/services/job-polling.service';

@Component({
  selector: 'qe-neural-net-dashboard',
  standalone: true,
  imports: [ReactiveFormsModule, JsonPipe],
  template: `
    <div class="qe-panel">
      <h2>🧠 Neural Networks (Week 3)</h2>
      <form [formGroup]="form">
        <div class="form-group">
          <label>Execution ID</label>
          <input formControlName="execution_id" type="text" />
        </div>
        <div class="form-group">
          <label>Model Type</label>
          <select formControlName="model_type">
            <option value="mlp">MLP</option>
            <option value="lstm">LSTM</option>
          </select>
        </div>
        <div class="form-group">
          <label>Epochs</label>
          <input formControlName="epochs" type="number" />
        </div>
        <div class="form-group">
          <label>Learning Rate</label>
          <input formControlName="learning_rate" type="number" step="0.0001" />
        </div>
        <div style="margin-top:16px">
          @if (currentAction()) {
            <span style="color:#a0aec0;font-size:12px;margin-right:12px">Action: {{ currentAction() }}</span>
          }
          <button type="button" [disabled]="loading()" (click)="submit('train')">
            {{ loading() && currentAction()==='Train' ? 'Running…' : 'Train' }}
          </button>
          <button type="button" [disabled]="loading()" (click)="submit('eval')">
            {{ loading() && currentAction()==='Evaluate' ? 'Running…' : 'Evaluate' }}
          </button>
          <button type="button" [disabled]="loading()" (click)="submit('predict')">
            {{ loading() && currentAction()==='Predict' ? 'Running…' : 'Predict' }}
          </button>
        </div>
      </form>
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
export class NeuralNetDashboardComponent {
  private readonly api     = inject(QuantEdgeAdminApiService);
  private readonly polling = inject(JobPollingService);
  private readonly fb      = inject(FormBuilder);

  readonly form = this.fb.group({
    execution_id:   ['nn-001'],
    model_type:     ['mlp'],
    epochs:         [5],
    learning_rate:  [0.001],
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

  submit(action: 'train' | 'eval' | 'predict'): void {
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);
    const v = this.form.value;
    const req = { execution_id: v.execution_id!, model_type: v.model_type!, epochs: v.epochs!, learning_rate: v.learning_rate! };

    const actionLabel = action === 'train' ? 'Train' : action === 'eval' ? 'Evaluate' : 'Predict';
    this.currentAction.set(actionLabel);

    const call$ = action === 'train' ? this.api.nnTrain(req)
                : action === 'eval'  ? this.api.nnEval(req)
                : this.api.nnPredict(req);

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
