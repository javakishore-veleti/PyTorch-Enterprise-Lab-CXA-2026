import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'qe-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <div class="qe-app">
      <header class="qe-topnav">
        <div class="nav-brand">
          <span class="logo">⬡</span>
          <span class="brand-name">QuantEdge</span>
        </div>
        <nav class="nav-links">
          <a routerLink="/home"      routerLinkActive="active">Home</a>
          <a routerLink="/forex"     routerLinkActive="active">💹 Forex</a>
          <a routerLink="/ai-chat"   routerLinkActive="active">🤖 AI Chat</a>
          <a routerLink="/inference" routerLinkActive="active">⚡ Inference</a>
          <a routerLink="/my-jobs"   routerLinkActive="active">📋 My Jobs</a>
        </nav>
      </header>
      <main class="qe-main">
        <router-outlet />
      </main>
    </div>
  `,
  styles: [`
    .qe-app { display: flex; flex-direction: column; min-height: 100vh; background: #0f1117; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    .qe-topnav { background: #1a1d27; border-bottom: 1px solid #2d3748; padding: 0 24px; display: flex; align-items: center; gap: 32px; height: 56px; }
    .nav-brand { display: flex; align-items: center; gap: 10px; }
    .logo { font-size: 24px; }
    .brand-name { font-weight: 700; font-size: 16px; color: #63b3ed; }
    .nav-links { display: flex; gap: 4px; }
    .nav-links a { color: #a0aec0; text-decoration: none; padding: 6px 14px; border-radius: 6px; font-size: 13px; transition: all 0.15s; }
    .nav-links a:hover { background: #2d3748; color: #e2e8f0; }
    .nav-links a.active { background: #2b4a7a; color: #63b3ed; }
    .qe-main { flex: 1; padding: 24px; max-width: 1200px; margin: 0 auto; width: 100%; box-sizing: border-box; }
  `],
})
export class AppComponent {
  readonly title = 'QuantEdge Customer Portal';
}
