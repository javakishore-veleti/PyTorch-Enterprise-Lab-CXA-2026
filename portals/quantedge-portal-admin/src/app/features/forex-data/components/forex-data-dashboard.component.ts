import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

/** Forex EUR/USD raw tick data viewer — Week 1 data pipeline (admin view). */
@Component({
  selector: 'qe-admin-forex-data',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div style="max-width:1100px">
      <div class="page-header">
        <h1>💹 Forex Pipeline</h1>
        <p>EUR/USD tick data ingestion, preprocessing &amp; autograd · Week 1</p>
      </div>
      <div class="qe-panel">
        <h2>EUR/USD Forex Tick Data</h2>
        <p style="color:#64748B">Raw HistData feed (~8 GB, 300M+ ticks). Use admin API to trigger ingestion and preprocessing.</p>
        <p style="color:#64748B">Endpoints: <code>POST /admin/foundations/forex/ingest</code> → <code>POST /admin/foundations/forex/preprocess</code></p>
      </div>
    </div>
  `,
})
export class ForexDataDashboardComponent {}
