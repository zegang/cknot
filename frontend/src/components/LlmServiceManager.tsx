import React, { useState, useEffect } from 'react';
import { getLlmServices, LLMService } from '../api/cknotApi';

interface LlmServiceManagerProps {
  token: string;
}

const LlmServiceManager: React.FC<LlmServiceManagerProps> = ({ token }) => {
  const [services, setServices] = useState<LLMService[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchServices = async () => {
    setLoading(true);
    setError('');
    try {
      const fetchedServices = await getLlmServices(token);
      setServices(fetchedServices);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch LLM services');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchServices();
  }, [token]);

  if (loading) return <p>Loading LLM services...</p>;
  if (error) return <p style={{ color: 'red' }}>Error: {error}</p>;

  return (
    <div style={{ maxWidth: '800px', margin: '20px auto', padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h2>LLM Service Manager</h2>
      <button onClick={fetchServices} style={{ marginBottom: '15px', padding: '8px 12px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
        Refresh Services
      </button>
      {services.length === 0 ? (
        <p>No LLM services registered. Use the API to add some.</p>
      ) : (
        <ul style={{ listStyleType: 'none', padding: 0 }}>
          {services.map((service) => (
            <li key={service.id} style={{ marginBottom: '10px', padding: '10px', border: '1px solid #eee', borderRadius: '5px', backgroundColor: '#fff' }}>
              <h3 style={{ margin: '0 0 5px 0', color: '#333' }}>{service.name} ({service.id})</h3>
              <p style={{ margin: '0' }}>Provider: {service.provider}</p>
              <p style={{ margin: '0' }}>Model: {service.model}</p>
              <p style={{ margin: '0' }}>Enabled: {service.is_enabled ? 'Yes' : 'No'}</p>
              <p style={{ margin: '0' }}>Valid: {service.is_valid ? 'Yes' : 'No'}</p>
              {/* Add buttons for enable/disable/test/delete here */}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default LlmServiceManager;