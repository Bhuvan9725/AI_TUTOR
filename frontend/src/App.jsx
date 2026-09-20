import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

import {
  Bot,
  ChevronDown,
  Copy,
  Menu,
  MessageSquare,
  Moon,
  Plus,
  RotateCcw,
  Send,
  Settings,
  Sparkles,
  Sun,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  User,
  X,
} from "lucide-react";

import "./App.css";

const API_URL = "https://ai-tutor-backend-qzbq.onrender.com";

const STORAGE_KEY = "ai_tutor_chats";

const suggestions = [
  "Explain artificial intelligence in simple words",
  "What is machine learning?",
  "Explain binary search with an example",
  "What is a neural network?",
];

const languages = [
  { value: "english", label: "English" },
  { value: "telugu", label: "తెలుగు" },
  { value: "hindi", label: "हिन्दी" },
];

const levels = [
  { value: "beginner", label: "Beginner" },
  { value: "intermediate", label: "Intermediate" },
  { value: "advanced", label: "Advanced" },
];

const modes = [
  { value: "tutor", label: "Tutor" },
  { value: "code tutor", label: "Code Tutor" },
];

function App() {
  const [chats, setChats] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [activeChatId, setActiveChatId] = useState(null);
  const [input, setInput] = useState("");

  const [language, setLanguage] = useState("english");
  const [level, setLevel] = useState("beginner");
  const [mode, setMode] = useState("tutor");

  const [darkMode, setDarkMode] = useState(true);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId),
    [chats, activeChatId]
  );

  const messages = activeChat?.messages || [];

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  }, [chats]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  const createNewChat = () => {
    const newChat = {
      id: Date.now().toString(),
      title: "New Chat",
      messages: [],
      createdAt: new Date().toISOString(),
    };

    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);
    setInput("");
    setSidebarOpen(false);
  };

  const deleteChat = (id) => {
    setChats((prev) => prev.filter((chat) => chat.id !== id));

    if (activeChatId === id) {
      setActiveChatId(null);
    }
  };

  const updateChat = (chatId, updater) => {
    setChats((prev) =>
      prev.map((chat) =>
        chat.id === chatId ? updater(chat) : chat
      )
    );
  };

  const startChatIfNeeded = () => {
    if (activeChatId) {
      return activeChatId;
    }

    const newChat = {
      id: Date.now().toString(),
      title: "New Chat",
      messages: [],
      createdAt: new Date().toISOString(),
    };

    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);

    return newChat.id;
  };

  const sendMessage = async (messageText = input) => {
    const text = messageText.trim();

    if (!text || loading) {
      return;
    }

    const chatId = startChatIfNeeded();

    const currentChat =
      chats.find((chat) => chat.id === chatId) || {
        id: chatId,
        title: "New Chat",
        messages: [],
      };

    const oldMessages = currentChat.messages || [];

    const userMessage = {
      id: Date.now().toString(),
      role: "user",
      content: text,
    };

    const assistantPlaceholder = {
      id: `${Date.now()}-assistant`,
      role: "assistant",
      content: "",
    };

    const updatedMessages = [
      ...oldMessages,
      userMessage,
      assistantPlaceholder,
    ];

    updateChat(chatId, (chat) => ({
      ...chat,
      title:
        chat.messages.length === 0
          ? text.length > 35
            ? `${text.slice(0, 35)}...`
            : text
          : chat.title,
      messages: updatedMessages,
    }));

    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: text,
          language,
          level,
          mode,
          history: oldMessages.map((item) => ({
            role: item.role,
            content: item.content,
          })),
        }),
      });

      /*
       * IMPORTANT:
       * If the backend returns 400, 404, 500, 502, etc.,
       * this will show the actual server error.
       */
      if (!response.ok) {
        const errorText = await response.text();

        throw new Error(
          `Server error ${response.status}: ${errorText}`
        );
      }

      const data = await response.json();

      const answer =
        data.answer ||
        data.response ||
        data.message ||
        "I couldn't generate a response.";

      updateChat(chatId, (chat) => ({
        ...chat,
        messages: chat.messages.map((message) =>
          message.id === assistantPlaceholder.id
            ? {
                ...message,
                content: answer,
              }
            : message
        ),
      }));
    } catch (error) {
      console.error("Chat error:", error);

      updateChat(chatId, (chat) => ({
        ...chat,
        messages: chat.messages.map((message) =>
          message.id === assistantPlaceholder.id
            ? {
                ...message,
                content: `Sorry, something went wrong.\n\n**Error:** ${error.message}`,
              }
            : message
        ),
      }));
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage();
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const copyMessage = async (content) => {
    try {
      await navigator.clipboard.writeText(content);
    } catch (error) {
      console.error("Copy failed:", error);
    }
  };

  const regenerateMessage = async (messageIndex) => {
    if (loading) {
      return;
    }

    const chat = chats.find((item) => item.id === activeChatId);

    if (!chat) {
      return;
    }

    const userMessage = [...chat.messages]
      .slice(0, messageIndex)
      .reverse()
      .find((message) => message.role === "user");

    if (!userMessage) {
      return;
    }

    const messagesWithoutAssistant = chat.messages.slice(
      0,
      messageIndex
    );

    updateChat(activeChatId, (current) => ({
      ...current,
      messages: messagesWithoutAssistant,
    }));

    setInput(userMessage.content);

    setTimeout(() => {
      sendMessage(userMessage.content);
    }, 100);
  };

  const rateMessage = (messageId, rating) => {
    updateChat(activeChatId, (chat) => ({
      ...chat,
      messages: chat.messages.map((message) =>
        message.id === messageId
          ? {
              ...message,
              rating,
            }
          : message
      ),
    }));
  };

  const handleSuggestion = (suggestion) => {
    sendMessage(suggestion);
  };

  return (
    <div className={darkMode ? "app dark" : "app"}>
      {/* SIDEBAR */}

      <aside className={sidebarOpen ? "sidebar open" : "sidebar"}>
        <div className="sidebar-header">
          <div className="brand">
            <div className="brand-icon">
              <Sparkles size={20} />
            </div>

            <div>
              <h2>AI Tutor</h2>
              <span>Multilingual Learning</span>
            </div>
          </div>

          <button
            className="icon-button mobile-close"
            onClick={() => setSidebarOpen(false)}
          >
            <X size={20} />
          </button>
        </div>

        <button className="new-chat-button" onClick={createNewChat}>
          <Plus size={18} />
          New Chat
        </button>

        <div className="recent-section">
          <div className="section-title">
            <span>Recent Chats</span>
          </div>

          <div className="chat-list">
            {chats.length === 0 ? (
              <div className="empty-chats">
                <MessageSquare size={18} />
                <span>No conversations yet</span>
              </div>
            ) : (
              chats.map((chat) => (
                <div
                  key={chat.id}
                  className={
                    chat.id === activeChatId
                      ? "chat-item active"
                      : "chat-item"
                  }
                  onClick={() => {
                    setActiveChatId(chat.id);
                    setSidebarOpen(false);
                  }}
                >
                  <MessageSquare size={17} />

                  <span className="chat-title">
                    {chat.title}
                  </span>

                  <button
                    className="delete-chat"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteChat(chat.id);
                    }}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="sidebar-control">
            <label>Language</label>

            <div className="select-wrapper">
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
              >
                {languages.map((item) => (
                  <option
                    key={item.value}
                    value={item.value}
                  >
                    {item.label}
                  </option>
                ))}
              </select>

              <ChevronDown size={15} />
            </div>
          </div>

          <div className="sidebar-control">
            <label>Level</label>

            <div className="select-wrapper">
              <select
                value={level}
                onChange={(e) => setLevel(e.target.value)}
              >
                {levels.map((item) => (
                  <option
                    key={item.value}
                    value={item.value}
                  >
                    {item.label}
                  </option>
                ))}
              </select>

              <ChevronDown size={15} />
            </div>
          </div>

          <div className="sidebar-control">
            <label>Mode</label>

            <div className="select-wrapper">
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
              >
                {modes.map((item) => (
                  <option
                    key={item.value}
                    value={item.value}
                  >
                    {item.label}
                  </option>
                ))}
              </select>

              <ChevronDown size={15} />
            </div>
          </div>

          <button
            className="theme-button"
            onClick={() => setDarkMode((prev) => !prev)}
          >
            {darkMode ? <Sun size={18} /> : <Moon size={18} />}

            <span>
              {darkMode ? "Light Mode" : "Dark Mode"}
            </span>
          </button>
        </div>
      </aside>

      {/* MOBILE OVERLAY */}

      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* MAIN */}

      <main className="main-content">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={22} />
          </button>

          <div className="topbar-title">
            <div className="topbar-icon">
              <Bot size={20} />
            </div>

            <div>
              <h1>AI Tutor</h1>
              <span>
                {language.charAt(0).toUpperCase() +
                  language.slice(1)}{" "}
                •{" "}
                {level.charAt(0).toUpperCase() +
                  level.slice(1)}
              </span>
            </div>
          </div>

          <button
            className="new-chat-top"
            onClick={createNewChat}
          >
            <Plus size={17} />
            <span>New Chat</span>
          </button>
        </header>

        <section className="chat-container">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-icon">
                <Sparkles size={30} />
              </div>

              <h2>How can I help you learn?</h2>

              <p>
                Ask me anything about programming, AI,
                machine learning, mathematics, or other
                subjects.
              </p>

              <div className="suggestions">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    className="suggestion-card"
                    onClick={() =>
                      handleSuggestion(suggestion)
                    }
                  >
                    <MessageSquare size={17} />

                    <span>{suggestion}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((message, index) => (
                <div
                  key={message.id}
                  className={
                    message.role === "user"
                      ? "message-row user-row"
                      : "message-row assistant-row"
                  }
                >
                  <div
                    className={
                      message.role === "user"
                        ? "avatar user-avatar"
                        : "avatar assistant-avatar"
                    }
                  >
                    {message.role === "user" ? (
                      <User size={17} />
                    ) : (
                      <Bot size={18} />
                    )}
                  </div>

                  <div className="message-content">
                    <div className="message-name">
                      {message.role === "user"
                        ? "You"
                        : "AI Tutor"}
                    </div>

                    <div className="message-bubble">
                      {message.content ? (
                        <ReactMarkdown
                          remarkPlugins={[
                            remarkGfm,
                            remarkMath,
                          ]}
                          rehypePlugins={[rehypeKatex]}
                        >
                          {message.content}
                        </ReactMarkdown>
                      ) : (
                        <div className="typing-indicator">
                          <span />
                          <span />
                          <span />
                        </div>
                      )}
                    </div>

                    {message.role === "assistant" &&
                      message.content && (
                        <div className="message-actions">
                          <button
                            onClick={() =>
                              copyMessage(message.content)
                            }
                            title="Copy"
                          >
                            <Copy size={15} />
                          </button>

                          <button
                            onClick={() =>
                              regenerateMessage(index)
                            }
                            title="Regenerate"
                          >
                            <RotateCcw size={15} />
                          </button>

                          <button
                            className={
                              message.rating === "up"
                                ? "rated"
                                : ""
                            }
                            onClick={() =>
                              rateMessage(
                                message.id,
                                "up"
                              )
                            }
                            title="Good response"
                          >
                            <ThumbsUp size={15} />
                          </button>

                          <button
                            className={
                              message.rating === "down"
                                ? "rated"
                                : ""
                            }
                            onClick={() =>
                              rateMessage(
                                message.id,
                                "down"
                              )
                            }
                            title="Bad response"
                          >
                            <ThumbsDown size={15} />
                          </button>
                        </div>
                      )}
                  </div>
                </div>
              ))}

              <div ref={messagesEndRef} />
            </div>
          )}
        </section>

        {/* INPUT */}

        <div className="input-area">
          <form
            className="input-box"
            onSubmit={handleSubmit}
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask AI Tutor anything..."
              rows={1}
              disabled={loading}
            />

            <button
              type="submit"
              className="send-button"
              disabled={!input.trim() || loading}
            >
              <Send size={18} />
            </button>
          </form>

          <div className="input-footer">
            <span>
              AI Tutor can make mistakes. Verify important
              information.
            </span>

            <span>
              Press <b>Enter</b> to send
            </span>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;