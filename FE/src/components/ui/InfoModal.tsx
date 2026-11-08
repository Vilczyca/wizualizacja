interface InfoModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function InfoModal({ isOpen, onClose }: InfoModalProps) {
  if (!isOpen) return null; // Jeśli isOpen to false, nie renderuj niczego

  return (
    <div style={{
      position: 'absolute',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      backgroundColor: '#fff',
      color: '#000',
      padding: '30px',
      borderRadius: '8px',
      boxShadow: '0 4px 15px rgba(0,0,0,0.3)',
      zIndex: 20,
      textAlign: 'center'
    }}>
      <h2>Hello, world</h2>
      <p>Tu będzie instrukcja</p>
      <button 
        onClick={onClose}
        style={{ marginTop: '20px', padding: '8px 16px', cursor: 'pointer' }}
      >
        Zamknij
      </button>
    </div>
  );
}