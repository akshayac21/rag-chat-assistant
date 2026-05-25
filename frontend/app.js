const API_URL = window.RAG_API_URL || "http://localhost:8000";
const SESSION_KEY = "rag_chat_session_id";

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");

function getSessionId() {
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
  }
  return sessionId;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(markdown) {
  const escaped = escapeHtml(markdown);
  const html = escaped
    .replace(/^### (.*)$/gm, "<strong>$1</strong>")
    .replace(/^## (.*)$/gm, "<strong>$1</strong>")
    .replace(/^# (.*)$/gm, "<strong>$1</strong>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
  return html
    .split("<br><br>")
    .map((paragraph) => `<p>${paragraph}</p>`)
    .join("");
}

function appendMessage(role, content, options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (options.rawHtml) {
    bubble.innerHTML = content;
  } else {
    bubble.innerHTML = renderMarkdown(content);
  }

  if (options.sources?.length) {
    const sources = document.createElement("div");
    sources.className = "sources";
    sources.textContent = `Sources: ${options.sources
      .map((source) => `${source.title} (${source.similarity})`)
      .join(", ")}`;
    bubble.appendChild(sources);
  }

  article.appendChild(bubble);
  messagesEl.appendChild(article);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return article;
}

function appendTypingIndicator() {
  return appendMessage(
    "assistant",
    '<span class="typing"><span></span><span></span><span></span></span>',
    { rawHtml: true },
  );
}

function typeAssistantMessage(content, sources) {
  const article = appendMessage("assistant", "");
  const bubble = article.querySelector(".bubble");
  let index = 0;

  const timer = window.setInterval(() => {
    index += 3;
    bubble.innerHTML = renderMarkdown(content.slice(0, index));
    messagesEl.scrollTop = messagesEl.scrollHeight;

    if (index >= content.length) {
      window.clearInterval(timer);
      bubble.innerHTML = renderMarkdown(content);
      if (sources?.length) {
        const sourcesEl = document.createElement("div");
        sourcesEl.className = "sources";
        sourcesEl.textContent = `Sources: ${sources
          .map((source) => `${source.title} (${source.similarity})`)
          .join(", ")}`;
        bubble.appendChild(sourcesEl);
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }, 12);
}

async function sendMessage(message) {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      session_id: getSessionId(),
    }),
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "The assistant could not process the request.");
  }
  return data;
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) {
    return;
  }

  appendMessage("user", message);
  inputEl.value = "";
  inputEl.focus();
  sendButton.disabled = true;
  const typing = appendTypingIndicator();

  try {
    const data = await sendMessage(message);
    typing.remove();
    typeAssistantMessage(data.answer, data.used_fallback ? [] : data.sources);
  } catch (error) {
    typing.remove();
    appendMessage("error", error.message || "Something went wrong.");
  } finally {
    sendButton.disabled = false;
  }
});
