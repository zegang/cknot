import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import AuthForm from './components/AuthForm';
import ChatInterface from './components/ChatInterface';
import LlmServiceManager from './components/LlmServiceManager';
import { setAuthToken, getAuthToken } from './api/cknotApi';

const App: React.FC = () => {
  const [token, setToken] = useState<string | null>(getAuthToken());
  const [username, setUsername] = useState<string | null>(localStorage.getItem('cknot_username'));

  useEffect(() => {
    if (token) {
      setAuthToken(token);
      localStorage.setItem('cknot_username', username || '');
    } else {
      setAuthToken(null);
      localStorage.removeItem('cknot_username');
    }
  }, [token, username]);

  const handleAuthSuccess = (newToken: string, newUsername: string) => {
    setToken(newToken);
    setUsername(newUsername);
  };

  const handleLogout = () => {
    setToken(null);
    setUsername(null);
  };

  return (
    <Router>
      <div style={{ fontFamily: 'Arial, sans-serif', padding: '20px' }}>
        <nav style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #eee', paddingBottom: '10px', marginBottom: '20px' }}>
          <h1 style={{ margin: 0, color: '#333' }}>CKnot UI</h1>
          <div>
            {token && username ? (
              <>
                <span style={{ marginRight: '15px' }}>Welcome, {username}!</span>
                <Link to="/chat" style={{ marginRight: '15px', textDecoration: 'none', color: '#007bff' }}>Chat</Link>
                <Link to="/llm-services" style={{ marginRight: '15px', textDecoration: 'none', color: '#007bff' }}>LLM Services</Link>
                <button onClick={handleLogout} style={{ padding: '8px 12px', backgroundColor: '#dc3545', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>Logout</button>
              </>
            ) : (
              <Link to="/login" style={{ textDecoration: 'none', color: '#007bff' }}>Login / Register</Link>
            )}
          </div>
        </nav>

        <Routes>
          <Route path="/login" element={token ? <Navigate to="/chat" /> : <AuthForm onAuthSuccess={handleAuthSuccess} />} />
          <Route path="/chat" element={token && username ? <ChatInterface token={token} username={username} /> : <Navigate to="/login" />} />
          <Route path="/llm-services" element={token ? <LlmServiceManager token={token} /> : <Navigate to="/login" />} />
          <Route path="/" element={<Navigate to={token ? "/chat" : "/login"} />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;