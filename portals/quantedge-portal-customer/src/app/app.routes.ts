import { Routes } from '@angular/router';

export const APP_ROUTES: Routes = [
  { path: '', redirectTo: 'forex', pathMatch: 'full' },
  {
    path: 'forex',
    loadComponent: () =>
      import('./features/forex/components/forex-dashboard.component')
        .then(m => m.ForexDashboardComponent),
    title: 'QuantEdge — Forex Signals',
  },
  {
    path: 'complaints',
    loadComponent: () =>
      import('./features/complaints/components/complaints-predict.component')
        .then(m => m.ComplaintsPredictComponent),
    title: 'QuantEdge — Complaint Classifier',
  },
  { path: '**', redirectTo: 'forex' },
];
