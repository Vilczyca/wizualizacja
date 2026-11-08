import { HiPlusCircle } from "react-icons/hi";
interface UploadButtonProps {
  onClick: () => void;
}

export function UploadButton({ onClick }: UploadButtonProps) {
  return (
    <button
      className="toolbar-button"
      onClick={onClick}
      aria-label="Upload"
      title="Upload"
    >
      <HiPlusCircle aria-hidden="true" />
    </button>
  );
}
