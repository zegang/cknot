import React, { useState, useEffect, useRef } from 'react';
import { sendMessage, approveAction, ChatResponse, ChatRequest } from '../api/cknotApi';

interface ChatInterfaceProps {
  token: string;
  username: string;
}

interface Message {
  sender: 'user' | 'cknot';
  content: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ token, username }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [sessionId, setSessionId] = useState<string>(`session-${Date.now()}-${username}`);
  const [isWaitingForApproval, setIsWaitingForApproval] = useState(false);
  const [nextNode, setNextNode] = useState<string | undefined>(undefined);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!inputMessage.trim() || isWaitingForApproval) return;

    const userMessage: Message = { sender: 'user', content: inputMessage };
    setMessages((prev) => [...prev, userMessage]);

    const chatRequest: ChatRequest = {
      message: inputMessage,
      session_id: sessionId,
      user_id: username,
    };

    setInputMessage('');
    try {
      const response = await sendMessage(token, chatRequest);
      handleAgentResponse(response);
    } catch (err: any) {
      setMessages((prev) => [...prev, { sender: 'cknot', content: `Error: ${err.response?.data?.detail || err.message}` }]);
      setIsWaitingForApproval(false);
    }
  };

  const handleAgentResponse = (response: ChatResponse) => {
    setMessages((prev) => [...prev, { sender: 'cknot', content: response.content }]);
    setIsWaitingForApproval(response.requires_action);
    setNextNode(response.next_node);
  };

  const handleApprove = async () => {
    if (!isWaitingForApproval) return;
    setIsWaitingForApproval(false); // Optimistically set to false
    try {
      const response = await approveAction(token, sessionId);
      handleAgentResponse(response);
    } catch (err: any) {
      setMessages((prev) => [...prev, { sender: 'cknot', content: `Approval Error: ${err.response?.data?.detail || err.message}` }]);
      setIsWaitingForApproval(false); // Revert if error
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 40px)', maxWidth: '800px', margin: '20px auto', border: '1px solid #ccc', borderRadius: '8px', overflow: 'hidden' }}>
      <div style={{ flexGrow: 1, padding: '15px', overflowY: 'auto', backgroundColor: '#f9f9f9' }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ marginBottom: '10px', textAlign: msg.sender === 'user' ? 'right' : 'left' }}>
            <span style={{
              display: 'inline-block',
              padding: '8px 12px',
              borderRadius: '18px',
              backgroundColor: msg.sender === 'user' ? '#007bff' : '#e0e0e0',
              color: msg.sender === 'user' ? 'white' : '#333',
              maxWidth: '70%',
              wordWrap: 'break-word',
            }}>
              {msg.content}
            </span>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div style={{ padding: '15px', borderTop: '1px solid #eee', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {isWaitingForApproval && (
          <div style={{ display: 'flex', justifyContent: 'center', gap: '10px' }}>
            <button onClick={handleApprove} style={{ padding: '10px 15px', backgroundColor: '#28a745', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
              Approve Action: {nextNode || 'Tool Call'}
            </button>
            <button onClick={() => setIsWaitingForApproval(false)} style={{ padding: '10px 15px', backgroundColor: '#dc3545', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
              Deny
            </button>
          </div>
        )}
        <form onSubmit={handleSendMessage} style={{ display: 'flex', gap: '10px' }}>
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder={isWaitingForApproval ? 'Approve or Deny action above' : 'Type your message...'}
            disabled={isWaitingForApproval}
            style={{ flexGrow: 1, padding: '10px', borderRadius: '5px', border: '1px solid #ccc' }}
          />
          <button type="submit" disabled={isWaitingForApproval} style={{ padding: '10px 15px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>Send</button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;