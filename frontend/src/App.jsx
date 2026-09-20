import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import {
  Plus,
  Send,
  Moon,
  Sun,
  Menu,
  Trash2,
  Copy,
  RefreshCw,
  ThumbsUp,
  ThumbsDown,
  Sparkles,
  MessageSquare,
  X
} from "lucide-react";

import "./App.css";

const API_URL = "https://ai-tutor-backend-qzbq.onrender.com";

const suggestions = [
  "What is Artificial Intelligence?",
  "Explain binary search with an example",
  "What is machine learning?",
  "Explain OOP concepts in Java"
];

function App() {
  const [chats, setChats] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("ai_tutor_chats")) || [];
    } catch {
      return [];
    }
  });

  const [activeChatId, setActiveChatId] = useState(null);
  const [message, setMessage] = useState("");
  const [language, setLanguage] = useState("english");
  const [level, setLevel] = useState("beginner");
  const [mode, setMode] = useState("tutor");
  const [darkMode, setDarkMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    localStorage.setItem("ai_tutor_chats", JSON.stringify(chats));
  }, [chats]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth"
    });
  }, [chats, activeChatId, loading]);

  const activeChat = chats.find(
    (chat) => chat.id === activeChatId
  );

  const messages = activeChat?.messages || [];

  const createNewChat = () => {
    setActiveChatId(null);
    setMessage("");
    setSidebarOpen(false);
  };

  const deleteChat = (id, event) => {
    event?.stopPropagation();

    setChats((prev) =>
      prev.filter((chat) => chat.id !== id)
    );

    if (activeChatId === id) {
      setActiveChatId(null);
    }
  };

  const updateChat = (chatId, newMessages) => {
    setChats((prev) =>
      prev.map((chat) =>
        chat.id === chatId
          ? {
              ...chat,
              messages: newMessages
            }
          : chat
      )
    );
  };

  const createChatIfNeeded = () => {
    if (activeChatId) {
      return activeChatId;
    }

    const id = Date.now().toString();

    const newChat = {
      id,
      title: "New Chat",
      messages: []
    };

    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(id);

    return id;
  };

  const sendMessage = async (customMessage = null) => {
    const text = (customMessage ?? message).trim();

    if (!text || loading) {
      return;
    }

    setMessage("");
    setLoading(true);

    const chatId = createChatIfNeeded();

    const currentChat =
      chats.find((chat) => chat.id === chatId);

    const oldMessages =
      currentChat?.messages || [];

    const userMessage = {
      id: Date.now(),
      role: "user",
      content: text
    };

    const updatedMessages = [
      ...oldMessages,
      userMessage
    ];

    setChats((prev) => {
      const exists = prev.some(
        (chat) => chat.id === chatId
      );

      if (!exists) {
        return [
          {
            id: chatId,
            title:
              text.length > 35
                ? text.substring(0, 35) + "..."
                : text,
            messages: updatedMessages
          },
          ...prev
        ];
      }

      return prev.map((chat) =>
        chat.id === chatId
          ? {
              ...chat,
              title:
                chat.title === "New Chat"
                  ? text.length > 35
                    ? text.substring(0, 35) + "..."
                    : text
                  : chat.title,
              messages: updatedMessages
            }
          : chat
      );
    });

    try {
      const response = await fetch(
        `${API_URL}/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            message: text,
            language,
            level,
            mode,
            history: oldMessages.map(
              (item) => ({
                role: item.role,
                content: item.content
              })
            )
          })
        }
      );

      if (!response.ok) {
        throw new Error(
          `Server error: ${response.status}`
        );
      }

      const data = await response.json();

      const assistantMessage = {
        id: Date.now() + 1,
        role: "assistant",
        content:
          data.answer ||
          data.response ||
          "Sorry, I could not generate a response."
      };

      setChats((prev) =>
        prev.map((chat) =>
          chat.id === chatId
            ? {
                ...chat,
                messages: [
                  ...updatedMessages,
                  assistantMessage
                ]
              }
            : chat
        )
      );
    } catch (error) {
      console.error(error);

      const errorMessage = {
        id: Date.now() + 1,
        role: "assistant",
        content:"Sorry, I couldn't connect to the AI Tutor backend. Please try again."
      };

      setChats((prev) =>
        prev.map((chat) =>
          chat.id === chatId
            ? {
                ...chat,
                messages: [
                  ...updatedMessages,
                  errorMessage
                ]
              }
            : chat
        )
      );
    } finally {
      setLoading(false);
    }
  };

  const regenerate = async (index) => {
    if (!activeChat || loading) {
      return;
    }

    const assistantMessage =
      activeChat.messages[index];

    if (
      !assistantMessage ||
      assistantMessage.role !== "assistant"
    ) {
      return;
    }

    const previousUserMessage =
      activeChat.messages[index - 1];

    if (
      !previousUserMessage ||
      previousUserMessage.role !== "user"
    ) {
      return;
    }

    const messagesBeforeAssistant =
      activeChat.messages.slice(0, index);

    updateChat(
      activeChat.id,
      messagesBeforeAssistant
    );

    setLoading(true);

    try {
      const history =
        messagesBeforeAssistant
          .slice(0, -1)
          .map((item) => ({
            role: item.role,
            content: item.content
          }));

      const response = await fetch(
        `${API_URL}/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            message: previousUserMessage.content,
            language,
            level,
            mode,
            history
          })
        }
      );

      if (!response.ok) {
        throw new Error("Failed to regenerate");
      }

      const data = await response.json();

      updateChat(activeChat.id, [
        ...messagesBeforeAssistant,
        {
          id: Date.now(),
          role: "assistant",
          content:
            data.answer ||
            data.response ||
            "Unable to regenerate response."
        }
      ]);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const copyMessage = async (content) => {
    try {
      await navigator.clipboard.writeText(content);
    } catch (error) {
      console.error(error);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  const selectSuggestion = (text) => {
    setMessage(text);

    setTimeout(() => {
      textareaRef.current?.focus();
    }, 50);
  };

  const formatTitle = (title) => {
    if (!title) {
      return "New Chat";
    }

    return title.length > 32
      ? title.substring(0, 32) + "..."
      : title;
  };

  return (
    <div
      className={`app ${
        darkMode ? "dark" : ""
      }`}
    >
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`sidebar ${
          sidebarOpen ? "open" : ""
        }`}
      >
        <div className="sidebar-brand">
          <div className="brand-icon">
            <Sparkles size={22} />
          </div>

          <div>
            <h2>AI Tutor</h2>
            <span>Adaptive Learning</span>
          </div>

          <button
            className="mobile-close"
            onClick={() =>
              setSidebarOpen(false)
            }
          >
            <X size={20} />
          </button>
        </div>

        <button
          className="new-chat-button"
          onClick={createNewChat}
        >
          <Plus size={20} />
          <span>New Chat</span>
        </button>

        <div className="recent-title">
          Recent Chats
        </div>

        <div className="chat-list">
          {chats.length === 0 ? (
            <div className="empty-chats">
              No recent chats
            </div>
          ) : (
            chats.map((chat) => (
              <div
                key={chat.id}
                className={`chat-item ${
                  activeChatId === chat.id
                    ? "active"
                    : ""
                }`}
                onClick={() => {
                  setActiveChatId(chat.id);
                  setSidebarOpen(false);
                }}
              >
                <MessageSquare size={18} />

                <span>
                  {formatTitle(chat.title)}
                </span>

                <button
                  className="delete-chat"
                  onClick={(event) =>
                    deleteChat(
                      chat.id,
                      event
                    )
                  }
                >
                  <Trash2 size={15} />
                </button>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-footer">
          <strong>AI Tutor</strong>
          <span>
            English • Telugu • Hindi
          </span>
        </div>
      </aside>

      {/* Main */}
      <main className="main">
        {/* Header */}
        <header className="header">
          <button
            className="menu-button"
            onClick={() =>
              setSidebarOpen(true)
            }
          >
            <Menu size={22} />
          </button>

          <div>
            <h1>
              Adaptive Multilingual AI Tutor
            </h1>

            <p>
              Learn • Practice • Understand
            </p>
          </div>

          <button
            className="theme-button"
            onClick={() =>
              setDarkMode((prev) => !prev)
            }
            title="Toggle theme"
          >
            {darkMode ? (
              <Sun size={21} />
            ) : (
              <Moon size={21} />
            )}
          </button>
        </header>

        {/* Controls */}
        <div className="controls">
          <div className="control-group">
            <label>Language</label>

            <select
              value={language}
              onChange={(event) =>
                setLanguage(event.target.value)
              }
            >
              <option value="english">
                English
              </option>

              <option value="telugu">
                Telugu
              </option>

              <option value="hindi">
                Hindi
              </option>
            </select>
          </div>

          <div className="control-group">
            <label>Level</label>

            <select
              value={level}
              onChange={(event) =>
                setLevel(event.target.value)
              }
            >
              <option value="beginner">
                Beginner
              </option>

              <option value="intermediate">
                Intermediate
              </option>

              <option value="advanced">
                Advanced
              </option>
            </select>
          </div>

          <div className="control-group">
            <label>Mode</label>

            <select
              value={mode}
              onChange={(event) =>
                setMode(event.target.value)
              }
            >
              <option value="tutor">
                Tutor
              </option>

              <option value="code tutor">
                Code Tutor
              </option>
            </select>
          </div>
        </div>

        {/* Chat */}
        <section className="chat-area">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon">
                <Sparkles size={30} />
              </div>

              <h2>
                What would you like to learn?
              </h2>

              <p>
                Ask me anything about programming,
                AI, machine learning, academics,
                or technical concepts.
              </p>

              <div className="suggestions">
                {suggestions.map(
                  (suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() =>
                        selectSuggestion(
                          suggestion
                        )
                      }
                    >
                      {suggestion}
                    </button>
                  )
                )}
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map(
                (item, index) => (
                  <div
                    key={item.id || index}
                    className={`message-row ${
                      item.role
                    }`}
                  >
                    <div
                      className={`avatar ${
                        item.role
                      }`}
                    >
                      {item.role ===
                      "user" ? (
                        "Y"
                      ) : (
                        <Sparkles size={18} />
                      )}
                    </div>

                    <div className="message-wrapper">
                      <div className="message-name">
                        {item.role ===
                        "user"
                          ? "You"
                          : "AI Tutor"}
                      </div>

                      <div className="message-content">
                        {item.role ===
                        "assistant" ? (
                          <ReactMarkdown
  remarkPlugins={[
    remarkGfm,
    remarkMath
  ]}
  rehypePlugins={[
    rehypeKatex
  ]}
>
  {item.content}
</ReactMarkdown>
                        ) : (
                          <p>
                            {item.content}
                          </p>
                        )}
                      </div>

                      {item.role ===
                        "assistant" && (
                        <div className="message-actions">
                          <button
                            onClick={() =>
                              copyMessage(
                                item.content
                              )
                            }
                            title="Copy"
                          >
                            <Copy size={15} />
                          </button>

                          <button
                            onClick={() =>
                              regenerate(index)
                            }
                            title="Regenerate"
                          >
                            <RefreshCw
                              size={15}
                            />
                          </button>

                          <button
                            title="Helpful"
                          >
                            <ThumbsUp
                              size={15}
                            />
                          </button>

                          <button
                            title="Not helpful"
                          >
                            <ThumbsDown
                              size={15}
                            />
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                )
              )}

              {loading && (
                <div className="message-row assistant">
                  <div className="avatar assistant">
                    <Sparkles size={18} />
                  </div>

                  <div className="message-wrapper">
                    <div className="message-name">
                      AI Tutor
                    </div>

                    <div className="typing">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </section>

        {/* Input */}
        <div className="input-section">
          <div className="input-box">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(event) =>
                setMessage(event.target.value)
              }
              onKeyDown={handleKeyDown}
              placeholder="Ask anything you want to learn..."
              rows={1}
              disabled={loading}
            />

            <button
              className="send-button"
              onClick={() =>
                sendMessage()
              }
              disabled={
                !message.trim() || loading
              }
              title="Send"
            >
              <Send size={20} />
            </button>
          </div>

          <p className="disclaimer">
            AI Tutor can make mistakes. Verify
            important information.
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;