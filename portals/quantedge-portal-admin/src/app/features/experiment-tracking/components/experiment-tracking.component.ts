import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

/** Experiment Tracking — connects to MLflow tracking server (Week 11). */
@Component({
  selector: 'qe-admin-experiment-tracking',
  standalone: true,
  imports: [CommonModule],
  template: `
    <h2>Experiment Tracking</h2>
    <p>MLflow tracking UI embedded here via backend <code>MLflowRegistryClient</code>.</p>
    <p>Tracking URI: <code>http://localhost:5000</code> — start with <code>npm run infra:up</code></p>
  `,
})
export class ExperimentTrackingComponent {}
