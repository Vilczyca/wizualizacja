interface InfoModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function InfoModal({ isOpen, onClose }: InfoModalProps) {
  if (!isOpen) return null; // Jeśli isOpen to false, nie renderuj niczego

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <section
        className="modal-panel info-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="info-title"
        onClick={(event) => event.stopPropagation()}
      >
        <p className="modal-eyebrow">Sterowanie</p>
        <h2 className="modal-title" id="info-title">
          Nawigacja po scenie
        </h2>
        <p className="modal-description">
          Użyj myszy, aby obejrzeć chmurę punktów z różnych stron.
        </p>

        <ul className="controls-list">
          <li>
            <span className="control-action">Obrót</span>
            <span className="control-input">
              Przeciągnij lewym przyciskiem myszy
            </span>
          </li>
          <li>
            <span className="control-action">Przesuwanie</span>
            <span className="control-input">
              Przeciągnij prawym przyciskiem
            </span>
          </li>
          <li>
            <span className="control-action">Przybliżanie</span>
            <span className="control-input">Użyj kółka myszy</span>
          </li>
        </ul>

        <div className="modal-footer">
          <button className="modal-close-button" onClick={onClose}>
            Zamknij
          </button>
        </div>
      </section>
    </div>
  );
}
