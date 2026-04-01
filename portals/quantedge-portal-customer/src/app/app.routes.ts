import { Routes } from '@angular/router';

export const APP_ROUTES: Routes = [
  { path: '', redirectTo: 'home', pathMatch: 'full' },
  { path: 'home', loadComponent: () => import('./features/home/components/home.component').then(m => m.HomeComponent), title: 'QuantEdge — Home' },
  { path: 'forex', loadComponent: () => import('./features/forex/components/forex-dashboard.component').then(m => m.ForexDashboardComponent), title: 'QuantEdge — Forex Signals' },
  { path: 'inference', loadComponent: () => import('./features/inference/components/model-inference.component').then(m => m.ModelInferenceComponent), title: 'QuantEdge — Inference' },
  { path: 'ai-chat', loadComponent: () => import('./features/ai-chat/components/ai-chat.component').then(m => m.AiChatComponent), title: 'QuantEdge — AI Chat' },
  { path: 'my-jobs', loadComponent: () => import('./features/my-jobs/components/my-jobs.component').then(m => m.MyJobsComponent), title: 'QuantEdge — My Jobs' },
  { path: 'complaints', loadComponent: () => import('./features/complaints/components/complaints-predict.component').then(m => m.ComplaintsPredictComponent), title: 'QuantEdge — Complaint Classifier' },
  { path: '**', redirectTo: 'home' },
];
