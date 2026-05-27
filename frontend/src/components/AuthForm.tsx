import React, { useState } from 'react';
import { login, register } from '../api/cknotApi';

interface AuthFormProps {
  onAuthSuccess: (token: string, username: string) => void;
}

const AuthForm: React.FC<AuthFormProps> = ({ onAuthSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isRegister, setIsRegister] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      if (isRegister) {
        await register({ username, password });
        alert('Registration successful! Please log in.');
        setIsRegister(false);
      } else {
        const response = await login(username, password);
        onAuthSuccess(response.access_token, username);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Authentication failed');
    }
  };

  return (
    <div style={{ maxWidth: '400px', margin: '50px auto', padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h2>{isRegister ? 'Register' : 'Login'} to CKnot</h2>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '15px' }}>
          <label htmlFor="username" style={{ display: 'block', marginBottom: '5px' }}>Username:</label>
          <input
            type="text"
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            style={{ width: '100%', padding: '8px', boxSizing: 'border-box' }}
          />
        </div>
        <div style={{ marginBottom: '15px' }}>
          <label htmlFor="password" style={{ display: 'block', marginBottom: '5px' }}>Password:</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{ width: '100%', padding: '8px', boxSizing: 'border-box' }}
          />
        </div>
        {error && <p style={{ color: 'red', marginBottom: '10px' }}>{error}</p>}
        <button type="submit" style={{ padding: '10px 15px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
          {isRegister ? 'Register' : 'Login'}
        </button>
        <button
          type="button"
          onClick={() => setIsRegister(!isRegister)}
          style={{ marginLeft: '10px', padding: '10px 15px', backgroundColor: '#6c757d', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}
        >
          {isRegister ? 'Go to Login' : 'Go to Register'}
        </button>
      </form>
    </div>
  );
};

export default AuthForm;