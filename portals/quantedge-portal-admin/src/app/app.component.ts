import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

interface NavItem { path: string; label: string; icon: string; week?: string; }

@Component({
  selector: 'qe-admin-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <div class="qe-shell">
      <aside class="qe-sidebar">
        <div class="brand">
          <span class="logo">⬡</span>
          <div>
            <div class="brand-name">QuantEdge</div>
            <div class="brand-sub">Admin · MLOps</div>
          </div>
        </div>
        <nav class="sidebar-nav">
          @for (item of navItems; track item.path) {
            <a [routerLink]="item.path" routerLinkActive="active" class="nav-item">
              <span class="nav-icon">{{ item.icon }}</span>
              <span class="nav-label">{{ item.label }}</span>
              @if (item.week) { <span class="nav-week">{{ item.week }}</span> }
            </a>
          }
        </nav>
      </aside>
      <main class="qe-content">
        <router-outlet />
      </main>
    </div>
  `,
  styles: [`
    .qe-shell { display: flex; height: 100vh; background: #0f1117; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    .qe-sidebar { width: 240px; background: #1a1d27; border-right: 1px solid #2d3748; display: flex; flex-direction: column; overflow-y: auto; }
    .brand { padding: 20px 16px; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid #2d3748; }
    .logo { font-size: 28px; }
    .brand-name { font-weight: 700; font-size: 16px; color: #63b3ed; }
    .brand-sub { font-size: 11px; color: #718096; }
    .sidebar-nav { padding: 8px 0; flex: 1; }
    .nav-item { display: flex; align-items: center; gap: 10px; padding: 9px 16px; text-decoration: none; color: #a0aec0; font-size: 13px; transition: all 0.15s; cursor: pointer; }
    .nav-item:hover { background: #2d3748; color: #e2e8f0; }
    .nav-item.active { background: #2b4a7a; color: #63b3ed; border-left: 3px solid #63b3ed; }
    .nav-icon { width: 20px; text-align: center; }
    .nav-label { flex: 1; }
    .nav-week { font-size: 10px; background: #2d3748; padding: 1px 6px; border-radius: 10px; color: #718096; }
    .qe-content { flex: 1; overflow-y: auto; padding: 24px; }
  `]
})
export class AppComponent {
  readonly title = 'QuantEdge Admin Portal';
  readonly navItems: NavItem[] = [
    { path: '/jobs',               label: 'All Jobs',             icon: '📋' },
    { path: '/forex',              label: 'Forex Pipeline',       icon: '💹', week: 'W1' },
    { path: '/banking-data',       label: 'CFPB Pipeline',        icon: '🏦', week: 'W2' },
    { path: '/neural-nets',        label: 'Neural Networks',      icon: '🧠', week: 'W3' },
    { path: '/profiling',          label: 'Profiling',            icon: '⚡', week: 'W4' },
    { path: '/attention',          label: 'Attention/Transformer',icon: '👁', week: 'W5-6' },
    { path: '/lora',               label: 'LoRA Fine-Tuning',     icon: '🔧', week: 'W7' },
    { path: '/domain-adapt',       label: 'Domain Adapt',         icon: '🤖', week: 'W8' },
    { path: '/export-formats',     label: 'Export Formats',       icon: '📦', week: 'W9' },
    { path: '/quantization',       label: 'Quantization',         icon: '🗜', week: 'W10' },
    { path: '/experiment-tracking',label: 'MLflow Tracking',      icon: '📊', week: 'W11' },
    { path: '/model-registry',     label: 'Canary Deploy',        icon: '🚦', week: 'W11' },
    { path: '/drift-monitoring',   label: 'Drift & Monitoring',   icon: '📡', week: 'W12' },
  ];
}
