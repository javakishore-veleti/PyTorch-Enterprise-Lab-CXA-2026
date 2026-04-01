import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'qe-admin-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <header class="qe-header admin">
      <div class="brand">
        <span class="logo">⬡</span>
        <span class="name">QuantEdge</span>
        <span class="tagline">Admin — MLOps Command</span>
      </div>
      <nav>
        <a routerLink="/training-jobs"       routerLinkActive="active">Training Jobs</a>
        <a routerLink="/banking-data"        routerLinkActive="active">Banking Data</a>
        <a routerLink="/experiment-tracking" routerLinkActive="active">Experiments</a>
        <a routerLink="/model-registry"      routerLinkActive="active">Model Registry</a>
      </nav>
    </header>
    <main class="qe-main">
      <router-outlet />
    </main>
  `,
})
export class AppComponent {
  readonly title = 'QuantEdge Admin Portal';
}
