import { Routes } from '@angular/router';

export const APP_ROUTES: Routes = [
  { path: '', redirectTo: 'jobs', pathMatch: 'full' },
  { path: 'jobs', loadComponent: () => import('./features/training-jobs/components/training-jobs-dashboard.component').then(m => m.TrainingJobsDashboardComponent), title: 'All Jobs' },
  { path: 'forex', loadComponent: () => import('./features/forex-data/components/forex-data-dashboard.component').then(m => m.ForexDataDashboardComponent), title: 'Forex Pipeline' },
  { path: 'banking-data', loadComponent: () => import('./features/banking-data/components/banking-data-dashboard.component').then(m => m.BankingDataDashboardComponent), title: 'CFPB Pipeline' },
  { path: 'neural-nets', loadComponent: () => import('./features/neural-nets/components/neural-net-dashboard.component').then(m => m.NeuralNetDashboardComponent), title: 'Neural Networks' },
  { path: 'profiling', loadComponent: () => import('./features/profiling/components/profiling-dashboard.component').then(m => m.ProfilingDashboardComponent), title: 'Profiling' },
  { path: 'attention', loadComponent: () => import('./features/attention/components/attention-dashboard.component').then(m => m.AttentionDashboardComponent), title: 'Attention' },
  { path: 'lora', loadComponent: () => import('./features/lora/components/lora-dashboard.component').then(m => m.LoraDashboardComponent), title: 'LoRA' },
  { path: 'domain-adapt', loadComponent: () => import('./features/domain-adapt/components/domain-adapt-dashboard.component').then(m => m.DomainAdaptDashboardComponent), title: 'Domain Adapt' },
  { path: 'export-formats', loadComponent: () => import('./features/export-formats/components/export-dashboard.component').then(m => m.ExportDashboardComponent), title: 'Export Formats' },
  { path: 'quantization', loadComponent: () => import('./features/quantization/components/quantization-dashboard.component').then(m => m.QuantizationDashboardComponent), title: 'Quantization' },
  { path: 'experiment-tracking', loadComponent: () => import('./features/experiment-tracking/components/experiment-tracking.component').then(m => m.ExperimentTrackingComponent), title: 'MLflow' },
  { path: 'model-registry', loadComponent: () => import('./features/model-registry/components/model-registry.component').then(m => m.ModelRegistryComponent), title: 'Canary' },
  { path: 'drift-monitoring', loadComponent: () => import('./features/drift-monitoring/components/drift-monitoring-dashboard.component').then(m => m.DriftMonitoringDashboardComponent), title: 'Drift & Monitoring' },
  { path: '**', redirectTo: 'jobs' },
];
