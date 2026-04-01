import { Routes } from '@angular/router';

export const APP_ROUTES: Routes = [
  { path: '', redirectTo: 'training-jobs', pathMatch: 'full' },
  {
    path: 'training-jobs',
    loadComponent: () =>
      import('./features/training-jobs/components/training-jobs-dashboard.component')
        .then(m => m.TrainingJobsDashboardComponent),
    title: 'QuantEdge Admin — Training Jobs',
  },
  {
    path: 'banking-data',
    loadComponent: () =>
      import('./features/banking-data/components/banking-data-dashboard.component')
        .then(m => m.BankingDataDashboardComponent),
    title: 'QuantEdge Admin — Banking Data',
  },
  {
    path: 'experiment-tracking',
    loadComponent: () =>
      import('./features/experiment-tracking/components/experiment-tracking.component')
        .then(m => m.ExperimentTrackingComponent),
    title: 'QuantEdge Admin — Experiments',
  },
  {
    path: 'model-registry',
    loadComponent: () =>
      import('./features/model-registry/components/model-registry.component')
        .then(m => m.ModelRegistryComponent),
    title: 'QuantEdge Admin — Model Registry',
  },
  { path: '**', redirectTo: 'training-jobs' },
];
