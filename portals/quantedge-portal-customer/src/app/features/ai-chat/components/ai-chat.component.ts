import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { QuantEdgeApiService } from '../../../core/services/quantedge-api.service';
import { JobPollingService } from '../../../core/services/job-polling.service';

interface ChatMessage { role: 'user' | 'assistant'; text: string; pending?: boolean; }

@Component({
  selector: 'qe-ai-chat',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="qe-panel chat-panel">
      <h2>🤖 AI Chat — Ollama (Llama3)</h2>
      <p style="color:#718096;font-size:13px;margin-bottom:16px">
        Financial Q&A, market analysis, and risk summaries powered by Ollama.
      </p>
      <div class="chat-messages">
        @for (msg of messages(); track $index) {
          <div class="chat-msg" [class.user-msg]="msg.role==='user'" [class.assistant-msg]="msg.role==='assistant'">
            <span class="msg-role">{{ msg.role === 'user' ? '🧑' : '🤖' }}</span>
            <div class="msg-body">
              @if (msg.pending) {
                <span class="typing">Thinking…</span>
              } @else {
                {{ msg.text }}
              }
            </div>
          </div>
        }
        @if (messages().length === 0) {
          <div class="chat-empty">Ask me about EUR/USD trends, risk metrics, or model performance.</div>
        }
      </div>
      <div class="chat-input-row">
        <input
          type="text"
          [(ngModel)]="inputText"
          (keydown.enter)="send()"
          placeholder="Ask about market data, models, or risk…"
          [disabled]="sending()"
          class="chat-input"
        />
        <button (click)="send()" [disabled]="sending() || !inputText.trim()">
          {{ sending() ? '…' : 'Send' }}
        </button>
      </div>
      @if (ollamaUnavailable()) {
        <div class="error-box" style="margin-top:12px">
          ⚠️ Ollama service unavailable. Start it with <code>ollama serve</code> and ensure a model is pulled.
        </div>
      }
    </div>
  `,
  styles: [`
    .chat-panel { display: flex; flex-direction: column; height: calc(100vh - 160px); }
    .chat-messages { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px; padding-right: 4px; }
    .chat-msg { display: flex; gap: 10px; align-items: flex-start; }
    .user-msg { flex-direction: row-reverse; }
    .msg-role { font-size: 20px; flex-shrink: 0; }
    .msg-body { background: #0d1117; border: 1px solid #2d3748; border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #e2e8f0; max-width: 70%; line-height: 1.5; }
    .user-msg .msg-body { background: #1a365d; border-color: #2b6cb0; }
    .typing { color: #718096; font-style: italic; }
    .chat-empty { color: #4a5568; text-align: center; padding: 40px; font-style: italic; }
    .chat-input-row { display: flex; gap: 8px; }
    .chat-input { flex: 1; background: #0d1117; border: 1px solid #2d3748; border-radius: 6px; padding: 10px 14px; color: #e2e8f0; font-size: 13px; }
    .chat-input:focus { outline: none; border-color: #63b3ed; }
  `],
})
export class AiChatComponent {
  private readonly api     = inject(QuantEdgeApiService);
  private readonly polling = inject(JobPollingService);

  messages         = signal<ChatMessage[]>([]);
  sending          = signal(false);
  ollamaUnavailable = signal(false);
  inputText        = '';

  send(): void {
    const text = this.inputText.trim();
    if (!text || this.sending()) return;

    this.inputText = '';
    this.sending.set(true);
    this.ollamaUnavailable.set(false);

    this.messages.update(msgs => [...msgs, { role: 'user', text }]);
    const pendingIdx = this.messages().length;
    this.messages.update(msgs => [...msgs, { role: 'assistant', text: '', pending: true }]);

    this.api.ollamaChat({ prompt: text, model_name: 'llama3' }).subscribe({
      next: (submitted) => {
        this.polling.pollUntilDone(submitted.job_id).subscribe({
          next: (status) => {
            if (status.status === 'ollama_unavailable' || status.error?.includes('ollama')) {
              this.ollamaUnavailable.set(true);
              this.messages.update(msgs => msgs.map((m, i) => i === pendingIdx ? { ...m, text: 'Ollama unavailable. Please start the Ollama service.', pending: false } : m));
              this.sending.set(false);
            } else if (status.result) {
              const responseText = typeof status.result === 'string' ? status.result
                : status.result?.response ?? status.result?.text ?? JSON.stringify(status.result);
              this.messages.update(msgs => msgs.map((m, i) => i === pendingIdx ? { ...m, text: responseText, pending: false } : m));
            }
          },
          error: (err) => {
            this.messages.update(msgs => msgs.map((m, i) => i === pendingIdx ? { ...m, text: `Error: ${err.message}`, pending: false } : m));
            this.sending.set(false);
          },
          complete: () => {
            this.messages.update(msgs => msgs.map(m => ({ ...m, pending: false })));
            this.sending.set(false);
          },
        });
      },
      error: (err) => {
        const isOllama = err.message?.toLowerCase().includes('ollama');
        if (isOllama) this.ollamaUnavailable.set(true);
        this.messages.update(msgs => msgs.map((m, i) => i === pendingIdx
          ? { ...m, text: isOllama ? 'Ollama unavailable.' : `Error: ${err.message}`, pending: false } : m));
        this.sending.set(false);
      },
    });
  }
}
