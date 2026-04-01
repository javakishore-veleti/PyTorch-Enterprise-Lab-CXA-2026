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
      <p style="color:#64748B;font-size:13px;margin-bottom:16px">
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
        <button class="send-btn" (click)="send()" [disabled]="sending() || !inputText.trim()">
          {{ sending() ? '…' : 'Send →' }}
        </button>
      </div>
      @if (ollamaUnavailable()) {
        <div class="ollama-warn">
          ⚠️ Ollama service unavailable. Start it with <code>ollama serve</code> and ensure a model is pulled.
        </div>
      }
    </div>
  `,
  styles: [`
    .chat-panel { display: flex; flex-direction: column; height: calc(100vh - 160px); background: #FFFFFF; border-radius: 16px; border: 1px solid #E2E8F0; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .chat-panel h2 { margin: 0 0 4px; font-size: 20px; font-weight: 700; color: #1E293B; }
    .chat-messages { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px; padding: 8px 4px 8px 0; }
    .chat-msg { display: flex; gap: 10px; align-items: flex-end; max-width: 85%; }
    .user-msg { flex-direction: row-reverse; align-self: flex-end; }
    .assistant-msg { align-self: flex-start; }
    .msg-role { font-size: 18px; flex-shrink: 0; margin-bottom: 2px; }
    .msg-body { border-radius: 18px; padding: 10px 16px; font-size: 13px; max-width: 100%; line-height: 1.6; }
    .user-msg .msg-body { background: linear-gradient(135deg, #2563EB, #7C3AED); color: #FFFFFF; border-bottom-right-radius: 4px; }
    .assistant-msg .msg-body { background: #FFFFFF; color: #1E293B; border: 1px solid #E2E8F0; border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .typing { color: #94A3B8; font-style: italic; }
    .chat-empty { color: #94A3B8; text-align: center; padding: 48px 20px; font-style: italic; font-size: 14px; }
    .chat-input-row { display: flex; gap: 8px; padding-top: 12px; border-top: 1px solid #F1F5F9; }
    .chat-input { flex: 1; background: #F8FAFC; border: 1.5px solid #E2E8F0; border-radius: 24px; padding: 10px 18px; color: #1E293B; font-size: 13px; font-family: inherit; transition: all 0.2s; }
    .chat-input:focus { outline: none; border-color: #2563EB; box-shadow: 0 0 0 3px rgba(37,99,235,0.12); background: #FFFFFF; }
    .chat-input:disabled { opacity: 0.6; }
    .send-btn { background: linear-gradient(135deg, #2563EB, #7C3AED); color: white; border: none; border-radius: 24px; padding: 10px 22px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; box-shadow: 0 4px 12px rgba(37,99,235,0.3); }
    .send-btn:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(37,99,235,0.4); }
    .send-btn:disabled { background: #E2E8F0; color: #94A3B8; cursor: not-allowed; box-shadow: none; transform: none; }
    .ollama-warn { background: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #D97706; border-radius: 10px; padding: 12px 16px; color: #92400E; font-size: 13px; margin-top: 12px; display: flex; align-items: center; gap: 8px; }
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
