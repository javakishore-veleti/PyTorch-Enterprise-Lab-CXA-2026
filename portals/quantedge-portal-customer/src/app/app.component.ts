import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'qe-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <header class="qe-header">
      <div class="brand">
        <span class="logo">⬡</span>
        <span class="name">QuantEdge</span>
        <span class="tagline">Financial AI Intelligence</span>
      </div>
      <nav>
        <a routerLink="/forex" routerLinkActive="active">Forex Signals</a>
        <a routerLink="/complaints" routerLinkActive="active">Complaint Classifier</a>
      </nav>
    </header>
    <main class="qe-main">
      <router-outlet />
    </main>
  `,
})
export class AppComponent {
  readonly title = 'QuantEdge Customer Portal';
}
