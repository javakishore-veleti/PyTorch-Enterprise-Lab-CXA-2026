import { Component, inject, signal, OnInit, OnDestroy } from '@angular/core';
import { JsonPipe, DatePipe } from '@angular/common';
import { QuantEdgeApiService } from '../../../core/services/quantedge-api.service';

interface JobRow { job_id: string; task_name: string; status: string; created_at: string; result?: any; expanded?: boolean; }

@Component({
  selector: 'qe-my-jobs',
  standalone: true,
  imports: [JsonPipe, DatePipe],
  template: `
    <div class="qe-panel">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
        <h2 style="margin:0">📋 My Jobs</h2>
        <span style="color:#718096;font-size:12px">Auto-refreshes every 10s</span>
      </div>

      @if (loading()) {
        <p style="color:#718096">Loading jobs…</p>
      } @else if (jobs().length === 0) {
        <p style="color:#4a5568;font-style:italic">No jobs found. Submit a request to see it here.</p>
      } @else {
        <table class="jobs-table">
          <thead>
            <tr>
              <th>Job ID</th>
              <th>Task</th>
              <th>Status</th>
              <th>Created</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            @for (job of jobs(); track job.job_id) {
              <tr (click)="toggleExpand(job)" style="cursor:pointer">
                <td><code style="font-size:11px">{{ job.job_id.slice(0,8) }}…</code></td>
                <td>{{ job.task_name }}</td>
                <td><span class="badge" [class]="statusClass(job.status)">{{ job.status }}</span></td>
                <td>{{ job.created_at | date:'HH:mm:ss' }}</td>
                <td>{{ job.result ? '▼ View' : '—' }}</td>
              </tr>
              @if (job.expanded && job.result) {
                <tr class="expand-row">
                  <td colspan="5">
                    <pre class="result-box">{{ job.result | json }}</pre>
                  </td>
                </tr>
              }
            }
          </tbody>
        </table>
      }
      @if (error()) {
        <div class="error-box">{{ error() }}</div>
      }
    </div>
  `,
  styles: [`
    .jobs-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .jobs-table th { text-align: left; background: #F8FAFC; color: #374151; font-weight: 700; padding: 10px 16px; border-bottom: 2px solid #E2E8F0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
    .jobs-table td { padding: 12px 16px; border-bottom: 1px solid #F1F5F9; color: #374151; }
    .jobs-table tr:hover td { background: #EFF6FF; }
    .expand-row td { background: #F8FAFC; }
  `],
})
export class MyJobsComponent implements OnInit, OnDestroy {
  private readonly api = inject(QuantEdgeApiService);
  private refreshTimer?: ReturnType<typeof setInterval>;

  jobs    = signal<JobRow[]>([]);
  loading = signal(false);
  error   = signal('');

  ngOnInit(): void {
    this.loadJobs();
    this.refreshTimer = setInterval(() => this.loadJobs(), 10_000);
  }

  ngOnDestroy(): void {
    if (this.refreshTimer) clearInterval(this.refreshTimer);
  }

  loadJobs(): void {
    this.loading.set(true);
    this.api.listJobs().subscribe({
      next: (resp: any) => {
        const jobList: JobRow[] = (resp?.jobs ?? []).map((j: any) => ({
          job_id:     j.job_id,
          task_name:  j.task_name,
          status:     j.status,
          created_at: j.created_at ?? j.submitted_at ?? '',
          result:     j.result ?? null,
          expanded:   false,
        }));
        this.jobs.set(jobList);
        this.loading.set(false);
      },
      error: (err) => { this.error.set(err.message); this.loading.set(false); },
    });
  }

  toggleExpand(job: JobRow): void {
    this.jobs.update(list => list.map(j => j.job_id === job.job_id ? { ...j, expanded: !j.expanded } : j));
  }

  statusClass(status: string): Record<string, boolean> {
    return {
      'badge-completed': status === 'completed' || status === 'success',
      'badge-failed':    status === 'failed',
      'badge-running':   status === 'running',
      'badge-pending':   status === 'pending',
    };
  }
}
