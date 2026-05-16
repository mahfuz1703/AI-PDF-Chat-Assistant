from langchain.callbacks.base import BaseCallbackHandler

CHAT_CSS = """
<style>
/* Hide Streamlit default header padding */
.block-container { padding-top: 1rem; }

.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-bottom: 80px;
}

.chat-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
}
.chat-row.user  { flex-direction: row-reverse; }

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.avatar.user      { background: #2563eb; }
.avatar.assistant { background: #10b981; }

.bubble {
    max-width: 70%;
    padding: 10px 16px;
    border-radius: 18px;
    line-height: 1.5;
    font-size: 15px;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.bubble.user {
    background: #2563eb;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}
.bubble.assistant {
    background: #f3f4f6;
    color: #111827;
    border-bottom-left-radius: 4px;
}
</style>
"""

class StreamHandler(BaseCallbackHandler):
    """Streams LLM tokens into a Streamlit placeholder as a chat bubble."""

    def __init__(self):
        self.container = None
        self.text = ""

    def reset(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        if self.container:
            safe = self.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            self.container.markdown(
                f'<div class="chat-row assistant">'
                f'<div class="avatar assistant">🤖</div>'
                f'<div class="bubble assistant">{safe}▌</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def build_messages_html(messages):
    parts = ['<div class="chat-wrapper">']
    for msg in messages:
        role = msg["role"]
        icon = "🧑" if role == "user" else "🤖"
        content = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(
            f'<div class="chat-row {role}">'
            f'<div class="avatar {role}">{icon}</div>'
            f'<div class="bubble {role}">{content}</div>'
            f'</div>'
        )
    parts.append("</div>")
    return "\n".join(parts)