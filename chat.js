(() => {
  const turns = [];
  const shell = document.createElement("div");
  shell.className = "assistant-shell";
  shell.hidden = true;
  shell.innerHTML = `
    <button class="assistant-scrim" type="button" aria-label="Close MarginShield assistant"></button>
    <aside class="assistant-panel" aria-labelledby="assistantTitle">
      <header class="assistant-head">
        <div><span>Grounded risk assistant</span><h2 id="assistantTitle">Ask MarginShield</h2></div>
        <button class="assistant-close" type="button" aria-label="Close assistant" title="Close">&#215;</button>
      </header>
      <div class="assistant-status"><i></i><span>Current model report, portfolio, and selected investigation context</span></div>
      <div class="assistant-transcript" role="log" aria-live="polite"></div>
      <form class="assistant-form">
        <label for="assistantInput">Question</label>
        <textarea id="assistantInput" rows="3" maxlength="2000" placeholder="Ask a grounded question"></textarea>
        <div><small class="assistant-provider">Local facts available</small><button type="submit">Send</button></div>
      </form>
    </aside>
  `;
  document.body.appendChild(shell);

  const transcript = shell.querySelector(".assistant-transcript");
  const form = shell.querySelector(".assistant-form");
  const input = shell.querySelector("#assistantInput");
  const submit = form.querySelector("button[type='submit']");
  const provider = shell.querySelector(".assistant-provider");

  function appendTurn(role, content) {
    const turn = document.createElement("article");
    turn.className = `assistant-turn ${role}`;
    const label = document.createElement("span");
    label.textContent = role === "assistant" ? "MarginShield" : "You";
    const text = document.createElement("p");
    text.textContent = content;
    turn.append(label, text);
    transcript.appendChild(turn);
    transcript.scrollTop = transcript.scrollHeight;
  }

  function openAssistant() {
    shell.hidden = false;
    document.body.classList.add("assistant-open");
    if (!turns.length) {
      appendTurn("assistant", "I answer from the current model report, portfolio, and selected case or ring. Unsupported claims are marked unavailable.");
    }
    window.setTimeout(() => input.focus(), 50);
  }

  function closeAssistant() {
    shell.hidden = true;
    document.body.classList.remove("assistant-open");
  }

  document.querySelectorAll("[data-chat-open]").forEach((button) => button.addEventListener("click", openAssistant));
  shell.querySelector(".assistant-close").addEventListener("click", closeAssistant);
  shell.querySelector(".assistant-scrim").addEventListener("click", closeAssistant);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !shell.hidden) closeAssistant();
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message || submit.disabled) return;
    const history = turns.slice(-8);
    turns.push({ role: "user", content: message });
    appendTurn("user", message);
    input.value = "";
    submit.disabled = true;
    submit.textContent = "Thinking";
    provider.textContent = "Checking grounded facts";

    try {
      const context = typeof window.marginShieldChatContext === "function"
        ? window.marginShieldChatContext()
        : { view: "casework", case_id: null, ring_id: null };
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, history, ...context }),
      });
      if (!response.ok) throw new Error("Assistant request failed");
      const result = await response.json();
      turns.push({ role: "assistant", content: result.answer });
      appendTurn("assistant", result.answer);
      provider.textContent = result.provider === "local_grounded_fallback" ? "Local grounded mode" : "Gemini free-tier mode";
    } catch (error) {
      appendTurn("assistant", "The grounded assistant is unavailable. No answer was generated.");
      provider.textContent = "Assistant unavailable";
    } finally {
      submit.disabled = false;
      submit.textContent = "Send";
      input.focus();
    }
  });
})();
