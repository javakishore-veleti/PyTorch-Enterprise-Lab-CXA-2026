import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

interface FeatureCard {
  title: string;
  description: string;
  icon: string;
  route: string;
  badge?: string;
}

@Component({
  selector: 'qe-home',
  standalone: true,
  imports: [RouterLink],
  template: `
    <div class="home-container">
      <div class="hero">
        <h1>⬡ QuantEdge</h1>
        <p class="hero-sub">Financial AI Intelligence Platform — Powered by PyTorch Enterprise Lab</p>
      </div>
      <div class="card-grid">
        @for (card of cards; track card.route) {
          <div class="feature-card">
            <div class="card-icon">{{ card.icon }}</div>
            <div class="card-body">
              <h3>{{ card.title }}</h3>
              @if (card.badge) {
                <span class="card-badge">{{ card.badge }}</span>
              }
              <p>{{ card.description }}</p>
              <a [routerLink]="card.route" class="card-btn">Open →</a>
            </div>
          </div>
        }
      </div>
    </div>
  `,
  styles: [`
    .home-container { max-width: 1100px; margin: 0 auto; }
    .hero { text-align: center; padding: 48px 0 32px; }
    .hero h1 { font-size: 42px; color: #63b3ed; margin: 0 0 12px; }
    .hero-sub { color: #718096; font-size: 16px; }
    .card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 20px; }
    .feature-card { background: #1a1d27; border: 1px solid #2d3748; border-radius: 10px; padding: 24px; display: flex; gap: 16px; transition: border-color 0.2s; }
    .feature-card:hover { border-color: #63b3ed; }
    .card-icon { font-size: 36px; }
    .card-body h3 { color: #e2e8f0; margin: 0 0 6px; font-size: 16px; }
    .card-body p { color: #718096; font-size: 13px; margin: 0 0 14px; line-height: 1.5; }
    .card-badge { background: #2b4a7a; color: #63b3ed; font-size: 10px; padding: 2px 8px; border-radius: 10px; margin-left: 6px; }
    .card-btn { display: inline-block; background: #2b6cb0; color: white; text-decoration: none; padding: 7px 16px; border-radius: 6px; font-size: 13px; }
    .card-btn:hover { background: #2c5282; }
  `],
})
export class HomeComponent {
  readonly cards: FeatureCard[] = [
    { title: 'Forex Signals', description: 'Live EUR/USD volatility and momentum signals powered by our PyTorch Transformer model.', icon: '💹', route: '/forex', badge: 'Live' },
    { title: 'AI Chat', description: 'Chat with Ollama LLM for financial Q&A, market summaries, and risk analysis.', icon: '🤖', route: '/ai-chat', badge: 'Llama3' },
    { title: 'Model Inference', description: 'Run direct inference against deployed PyTorch models in various export formats.', icon: '⚡', route: '/inference' },
    { title: 'My Jobs', description: 'Track all your submitted jobs with real-time status updates and result inspection.', icon: '📋', route: '/my-jobs' },
    { title: 'Market Alerts', description: 'Configure drift thresholds and receive alerts when model predictions shift.', icon: '📡', route: '/forex' },
  ];
}
