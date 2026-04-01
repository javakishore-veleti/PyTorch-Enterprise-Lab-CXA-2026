import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

/** Forex EUR/USD raw tick data viewer — Week 1 data pipeline (admin view). */
@Component({
  selector: 'qe-admin-forex-data',
  standalone: true,
  imports: [CommonModule],
  template: `
    <h2>EUR/USD Forex Tick Data</h2>
    <p>Raw HistData feed (~8 GB, 300M+ ticks). Use admin API to trigger ingestion and preprocessing.</p>
    <p>Endpoints: <code>POST /admin/foundations/forex/ingest</code> → <code>POST /admin/foundations/forex/preprocess</code></p>
  `,
})
export class ForexDataDashboardComponent {}
