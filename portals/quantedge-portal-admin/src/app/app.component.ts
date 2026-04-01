import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

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
            <div class="brand-sub">Admin · MLOps Command</div>
          </div>
        </div>
        <nav class="sidebar-nav">
          <div class="sidebar-section">Operations</div>
          <a routerLink="/jobs"           routerLinkActive="active" class="nav-item"><span class="nav-icon">📋</span><span class="nav-label">All Jobs</span></a>
          <a routerLink="/forex"          routerLinkActive="active" class="nav-item"><span class="nav-icon">💹</span><span class="nav-label">Forex Pipeline</span><span class="nav-week">W1</span></a>
          <a routerLink="/banking-data"   routerLinkActive="active" class="nav-item"><span class="nav-icon">🏦</span><span class="nav-label">CFPB Pipeline</span><span class="nav-week">W2</span></a>
          <div class="sidebar-section">Model Training</div>
          <a routerLink="/neural-nets"    routerLinkActive="active" class="nav-item"><span class="nav-icon">🧠</span><span class="nav-label">Neural Networks</span><span class="nav-week">W3</span></a>
          <a routerLink="/profiling"      routerLinkActive="active" class="nav-item"><span class="nav-icon">⚡</span><span class="nav-label">Profiling</span><span class="nav-week">W4</span></a>
          <a routerLink="/attention"      routerLinkActive="active" class="nav-item"><span class="nav-icon">👁</span><span class="nav-label">Attention/Transformer</span><span class="nav-week">W5-6</span></a>
          <a routerLink="/lora"           routerLinkActive="active" class="nav-item"><span class="nav-icon">🔧</span><span class="nav-label">LoRA Fine-Tuning</span><span class="nav-week">W7</span></a>
          <a routerLink="/domain-adapt"   routerLinkActive="active" class="nav-item"><span class="nav-icon">🤖</span><span class="nav-label">Domain Adapt</span><span class="nav-week">W8</span></a>
          <div class="sidebar-section">Deployment</div>
          <a routerLink="/export-formats" routerLinkActive="active" class="nav-item"><span class="nav-icon">📦</span><span class="nav-label">Export Formats</span><span class="nav-week">W9</span></a>
          <a routerLink="/quantization"   routerLinkActive="active" class="nav-item"><span class="nav-icon">🗜</span><span class="nav-label">Quantization</span><span class="nav-week">W10</span></a>
          <div class="sidebar-section">MLOps</div>
          <a routerLink="/experiment-tracking" routerLinkActive="active" class="nav-item"><span class="nav-icon">📊</span><span class="nav-label">MLflow Tracking</span><span class="nav-week">W11</span></a>
          <a routerLink="/model-registry"      routerLinkActive="active" class="nav-item"><span class="nav-icon">🚦</span><span class="nav-label">Canary Deploy</span><span class="nav-week">W11</span></a>
          <a routerLink="/drift-monitoring"    routerLinkActive="active" class="nav-item"><span class="nav-icon">📡</span><span class="nav-label">Drift & Monitoring</span><span class="nav-week">W12</span></a>
        </nav>
        <div class="sidebar-footer">QuantEdge CXA-2026 · v1.0</div>
      </aside>
      <main class="qe-content">
        <router-outlet />
      </main>
    </div>
  `,
  styles: [`
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .qe-shell { display: flex; height: 100vh; background: #F1F5F9; font-family: 'Inter', sans-serif; }

    .qe-sidebar {
      width: 256px;
      background: linear-gradient(180deg, #1E1B4B 0%, #312E81 30%, #4338CA 70%, #4F46E5 100%);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      box-shadow: 4px 0 24px rgba(79,70,229,0.25);
      flex-shrink: 0;
    }

    .brand {
      padding: 24px 20px;
      display: flex;
      align-items: center;
      gap: 12px;
      border-bottom: 1px solid rgba(255,255,255,0.12);
    }
    .logo { font-size: 32px; filter: drop-shadow(0 0 8px rgba(196,181,253,0.6)); }
    .brand-name { font-weight: 800; font-size: 18px; color: #FFFFFF; letter-spacing: -0.3px; }
    .brand-sub { font-size: 11px; color: rgba(196,181,253,0.8); font-weight: 500; }

    .sidebar-section { padding: 12px 12px 4px; font-size: 10px; font-weight: 700; color: rgba(196,181,253,0.55); text-transform: uppercase; letter-spacing: 0.1em; }

    .sidebar-nav { padding: 8px 12px; flex: 1; }
    .nav-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 8px;
      text-decoration: none;
      color: rgba(196,181,253,0.85);
      font-size: 13px;
      font-weight: 500;
      transition: all 0.15s;
      margin-bottom: 2px;
    }
    .nav-item:hover { background: rgba(255,255,255,0.12); color: #FFFFFF; }
    .nav-item.active {
      background: rgba(255,255,255,0.18);
      color: #FFFFFF;
      font-weight: 600;
      box-shadow: inset 3px 0 0 #A5B4FC;
    }
    .nav-icon { width: 22px; text-align: center; font-size: 16px; }
    .nav-label { flex: 1; }
    .nav-week {
      font-size: 9px;
      background: rgba(255,255,255,0.15);
      color: rgba(196,181,253,0.9);
      padding: 2px 7px;
      border-radius: 10px;
      font-weight: 600;
    }

    .sidebar-footer {
      padding: 16px 20px;
      border-top: 1px solid rgba(255,255,255,0.12);
      font-size: 11px;
      color: rgba(196,181,253,0.5);
      text-align: center;
    }

    .qe-content {
      flex: 1;
      overflow-y: auto;
      padding: 28px 32px;
      background: #F1F5F9;
    }
  `]
})
export class AppComponent {
  readonly title = 'QuantEdge Admin Portal';
}
