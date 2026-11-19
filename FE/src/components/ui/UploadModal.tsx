interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function UploadModal({ isOpen, onClose }: UploadModalProps) {
  if (!isOpen) return null; // Jeśli isOpen to false, nie renderuj niczego

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <section
        className="modal-panel upload-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-title"
        onClick={(event) => event.stopPropagation()}
      >
        <p className="modal-eyebrow">Import danych</p>
        <h2 className="modal-title" id="upload-title">
          Dodaj dane
        </h2>
        <p className="modal-description">TODO</p>

        {/* <div className="upload-placeholder">
          <span className="upload-placeholder-icon" aria-hidden="true">
            +
          </span>
          <span className="upload-placeholder-title">Obszar dodawania plików</span>
          <span className="upload-placeholder-description">
            Obsługa wczytywania plików będzie dostępna wkrótce.
          </span>
        </div> */}

        <div className="modal-footer">
          <button className="modal-close-button" onClick={onClose}>
            Zamknij
          </button>
        </div>
      </section>
    </div>
  );
}
