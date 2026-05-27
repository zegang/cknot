import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// --- Authentication ---
export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface UserRegisterRequest {
  username: string;
  password: string;
  email?: string;
}

export const login = async (username: string, password: string): Promise<TokenResponse> => {
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);

  const response = await apiClient.post<TokenResponse>('/token', formData, {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
  });
  return response.data;
};

export const register = async (data: UserRegisterRequest): Promise<{ message: string }> => {
  const response = await apiClient.post<{ message: string }>('/register', data);
  return response.data;
};

// --- LLM Service Management ---
export interface LLMService {
  id: string;
  name: string;
  provider: string;
  model: string;
  api_key?: string;
  base_url?: string;
  is_enabled: boolean;
  is_valid: boolean;
}

export const getLlmServices = async (token: string): Promise<LLMService[]> => {
  const response = await apiClient.get<LLMService[]>('/llm_services', {
    headers: { Authorization: `Bearer ${token}` },
  });
  return response.data;
};

export const registerLlmService = async (token: string, service: LLMService): Promise<{ message: string }> => {
  const response = await apiClient.post<{ message: string }>('/llm_services', service, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return response.data;
};

// --- Chat Interaction ---
export interface ChatRequest {
  message: string;
  session_id: string;
  user_id: string;
  current_task?: string;
}

export interface ChatResponse {
  content: string;
  session_id: string;
  requires_action: boolean;
  next_node?: string;
}

export const sendMessage = async (token: string, chatRequest: ChatRequest): Promise<ChatResponse> => {
  const response = await apiClient.post<ChatResponse>('/chat', chatRequest, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return response.data;
};

export const approveAction = async (token: string, sessionId: string): Promise<ChatResponse> => {
  const response = await apiClient.post<ChatResponse>(`/approve/${sessionId}`, {}, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return response.data;
};

// --- Utility to set/get token ---
export const setAuthToken = (token: string | null) => {
  if (token) {
    localStorage.setItem('cknot_auth_token', token);
  } else {
    localStorage.removeItem('cknot_auth_token');
  }
};

export const getAuthToken = (): string | null => {
  return localStorage.getItem('cknot_auth_token');
};