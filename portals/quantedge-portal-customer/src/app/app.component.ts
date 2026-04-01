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
          <div>
            <span class="brand-name">QuantEdge</span>
            <span class="brand-tagline">Financial AI Platform</span>
          </div>
        </div>
        <nav class="nav-links">
          <a routerLink="/home"      routerLinkActive="active" [routerLinkActiveOptions]="{exact:true}">🏠 Home</a>
          <a routerLink="/forex"     routerLinkActive="active">💹 Forex</a>
          <a routerLink="/ai-chat"   routerLinkActive="active">🤖 AI Chat</a>
          <a routerLink="/inference" routerLinkActive="active">⚡ Inference</a>
          <a routerLink="/my-jobs"   routerLinkActive="active">📋 My Jobs</a>
        </nav>
        <div class="nav-status">
          <span class="status-dot"></span>
          <span class="status-text">Live</span>
        </div>
      </header>
      <main class="qe-main">
        <router-outlet />
      </main>
    </div>
  `,
  styles: [`
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .qe-app { display: flex; flex-direction: column; min-height: 100vh; background: #F8FAFC; font-family: 'Inter', sans-serif; }

    .qe-topnav {
      background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 40%, #2563EB 70%, #7C3AED 100%);
      padding: 0 32px;
      display: flex;
      align-items: center;
      gap: 24px;
      height: 64px;
      box-shadow: 0 4px 20px rgba(37,99,235,0.35);
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .nav-brand { display: flex; align-items: center; gap: 12px; margin-right: 8px; }
    .logo { font-size: 28px; filter: drop-shadow(0 0 8px rgba(167,243,208,0.5)); }
    .brand-name { font-weight: 800; font-size: 18px; color: #FFFFFF; display: block; line-height: 1.2; letter-spacing: -0.3px; }
    .brand-tagline { font-size: 10px; color: rgba(186,230,253,0.85); font-weight: 500; display: block; }

    .nav-links { display: flex; gap: 4px; flex: 1; }
    .nav-links a {
      color: rgba(186,230,253,0.9);
      text-decoration: none;
      padding: 7px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 500;
      transition: all 0.2s;
    }
    .nav-links a:hover { background: rgba(255,255,255,0.15); color: #FFFFFF; }
    .nav-links a.active { background: rgba(255,255,255,0.2); color: #FFFFFF; font-weight: 600; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }

    .nav-status { display: flex; align-items: center; gap: 6px; }
    .status-dot { width: 8px; height: 8px; background: #10B981; border-radius: 50%; box-shadow: 0 0 6px #10B981; animation: pulse-green 2s infinite; }
    .status-text { color: rgba(186,230,253,0.9); font-size: 12px; font-weight: 500; }
    @keyframes pulse-green { 0%,100%{opacity:1} 50%{opacity:0.5} }

    .qe-main { flex: 1; padding: 28px 32px; max-width: 1200px; margin: 0 auto; width: 100%; box-sizing: border-box; }
  `],
})
export class AppComponent {
  readonly title = 'QuantEdge Customer Portal';
}
