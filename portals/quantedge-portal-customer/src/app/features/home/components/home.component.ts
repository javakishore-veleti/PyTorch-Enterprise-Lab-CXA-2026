import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'qe-home',
  standalone: true,
  imports: [RouterLink],
  template: `
    <div class="home-wrap">
      <!-- Hero -->
      <div class="hero-banner">
        <div class="hero-content">
          <div class="hero-badge">🚀 Powered by PyTorch Enterprise Lab</div>
          <h1 class="hero-title">Financial AI<br><span class="hero-gradient">Intelligence Platform</span></h1>
          <p class="hero-desc">Enterprise-grade AI models for forex prediction, NLP, and market analysis — built on 12 weeks of cutting-edge PyTorch research.</p>
          <div class="hero-actions">
            <a routerLink="/forex" class="hero-btn-primary">Get Forex Signals →</a>
            <a routerLink="/ai-chat" class="hero-btn-secondary">Try AI Chat</a>
          </div>
        </div>
        <div class="hero-visual">
          <div class="hero-chart">
            <div class="chart-bar" style="height:60%"></div>
            <div class="chart-bar" style="height:80%"></div>
            <div class="chart-bar" style="height:55%"></div>
            <div class="chart-bar" style="height:90%"></div>
            <div class="chart-bar" style="height:70%"></div>
            <div class="chart-bar" style="height:95%"></div>
            <div class="chart-bar active" style="height:85%"></div>
          </div>
        </div>
      </div>

      <!-- Stats row -->
      <div class="stats-strip">
        <div class="stat-item"><div class="stat-num">12</div><div class="stat-lbl">Weeks of Training</div></div>
        <div class="stat-item"><div class="stat-num">248</div><div class="stat-lbl">Passing Tests</div></div>
        <div class="stat-item"><div class="stat-num">40+</div><div class="stat-lbl">API Endpoints</div></div>
        <div class="stat-item"><div class="stat-num">6</div><div class="stat-lbl">Model Formats</div></div>
      </div>

      <!-- Feature cards -->
      <h2 class="section-title">Explore Platform Features</h2>
      <div class="card-grid">
        @for (card of cards; track card.route) {
          <a [routerLink]="card.route" class="feature-card">
            <div class="card-header" [style.background]="card.gradient">
              <span class="card-icon">{{ card.icon }}</span>
              @if (card.badge) { <span class="card-badge-pill">{{ card.badge }}</span> }
            </div>
            <div class="card-body">
              <h3>{{ card.title }}</h3>
              <p>{{ card.description }}</p>
              <span class="card-cta">Open {{ card.title }} →</span>
            </div>
          </a>
        }
      </div>
    </div>
  `,
  styles: [`
    .home-wrap { max-width: 1100px; margin: 0 auto; }

    /* Hero */
    .hero-banner {
      background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 50%, #7C3AED 100%);
      border-radius: 20px;
      padding: 52px 48px;
      margin-bottom: 32px;
      display: flex;
      align-items: center;
      gap: 48px;
      overflow: hidden;
      position: relative;
      box-shadow: 0 20px 40px rgba(37,99,235,0.3);
    }
    .hero-banner::after {
      content:''; position:absolute; top:-60px; right:240px; width:300px; height:300px;
      border-radius:50%; background:rgba(255,255,255,0.04);
    }
    .hero-content { flex: 1; position: relative; z-index: 1; }
    .hero-badge { display:inline-block; background:rgba(255,255,255,0.15); color:rgba(186,230,253,0.95); padding:5px 14px; border-radius:20px; font-size:12px; font-weight:600; margin-bottom:16px; }
    .hero-title { font-size: 42px; font-weight: 800; color: #FFFFFF; margin: 0 0 16px; line-height: 1.15; letter-spacing: -1px; }
    .hero-gradient { background: linear-gradient(90deg, #67E8F9, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .hero-desc { color: rgba(186,230,253,0.85); font-size: 15px; line-height: 1.7; margin: 0 0 28px; max-width: 480px; }
    .hero-actions { display: flex; gap: 12px; flex-wrap: wrap; }
    .hero-btn-primary { background: #FFFFFF; color: #1D4ED8; padding: 12px 24px; border-radius: 10px; text-decoration: none; font-weight: 700; font-size: 14px; transition: all 0.2s; box-shadow: 0 4px 14px rgba(0,0,0,0.15); }
    .hero-btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.2); }
    .hero-btn-secondary { background: rgba(255,255,255,0.15); color: #FFFFFF; padding: 12px 24px; border-radius: 10px; text-decoration: none; font-weight: 600; font-size: 14px; border: 1.5px solid rgba(255,255,255,0.3); transition: all 0.2s; }
    .hero-btn-secondary:hover { background: rgba(255,255,255,0.25); }

    /* Chart visual */
    .hero-visual { flex-shrink: 0; }
    .hero-chart { display: flex; align-items: flex-end; gap: 8px; height: 120px; background: rgba(255,255,255,0.08); border-radius: 12px; padding: 16px 20px; }
    .chart-bar { width: 20px; background: rgba(103,232,249,0.5); border-radius: 4px 4px 0 0; transition: height 0.3s; }
    .chart-bar.active { background: #67E8F9; box-shadow: 0 0 12px rgba(103,232,249,0.6); }

    /* Stats */
    .stats-strip { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 40px; }
    .stat-item { background: white; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .stat-num { font-size: 32px; font-weight: 800; background: linear-gradient(135deg, #2563EB, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .stat-lbl { font-size: 12px; color: #64748B; font-weight: 500; margin-top: 4px; }

    /* Section */
    .section-title { font-size: 22px; font-weight: 700; color: #1E293B; margin: 0 0 20px; }

    /* Cards */
    .card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; margin-bottom: 32px; }
    .feature-card { background: white; border-radius: 16px; overflow: hidden; text-decoration: none; color: inherit; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); transition: all 0.25s; display: block; }
    .feature-card:hover { transform: translateY(-4px); box-shadow: 0 12px 30px rgba(37,99,235,0.15); border-color: #BFDBFE; }
    .card-header { padding: 28px 24px 20px; display: flex; justify-content: space-between; align-items: flex-start; }
    .card-icon { font-size: 36px; }
    .card-badge-pill { background: rgba(255,255,255,0.25); color: white; font-size: 10px; font-weight: 700; padding: 3px 10px; border-radius: 20px; backdrop-filter: blur(4px); }
    .card-body { padding: 0 24px 24px; }
    .card-body h3 { font-size: 17px; font-weight: 700; color: #1E293B; margin: 0 0 8px; }
    .card-body p { font-size: 13px; color: #64748B; line-height: 1.6; margin: 0 0 16px; }
    .card-cta { font-size: 13px; font-weight: 600; color: #2563EB; }
  `],
})
export class HomeComponent {
  readonly cards = [
    { title: 'Forex Signals', description: 'Live EUR/USD volatility and momentum signals powered by our PyTorch Transformer model.', icon: '💹', route: '/forex', badge: 'Live', gradient: 'linear-gradient(135deg,#059669,#0D9488)' },
    { title: 'AI Chat', description: 'Chat with Ollama LLM for financial Q&A, market summaries, and risk analysis.', icon: '🤖', route: '/ai-chat', badge: 'Llama3', gradient: 'linear-gradient(135deg,#7C3AED,#6D28D9)' },
    { title: 'Model Inference', description: 'Run direct inference against deployed PyTorch models in TorchScript, ONNX, or quantized formats.', icon: '⚡', route: '/inference', gradient: 'linear-gradient(135deg,#D97706,#B45309)' },
    { title: 'My Jobs', description: 'Track all your submitted jobs with real-time status updates and result inspection.', icon: '📋', route: '/my-jobs', gradient: 'linear-gradient(135deg,#2563EB,#1D4ED8)' },
    { title: 'Market Alerts', description: 'Monitor drift thresholds and get alerted when model predictions shift significantly.', icon: '��', route: '/forex', badge: 'Beta', gradient: 'linear-gradient(135deg,#DC2626,#B91C1C)' },
  ];
}
