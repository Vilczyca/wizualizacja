import { HiQuestionMarkCircle } from "react-icons/hi";
interface InfoButtonProps {
  onClick: () => void;
}

export function InfoButton({ onClick }: InfoButtonProps) {
  return (
    <button className="toolbar-button" onClick={onClick} aria-label="Info" title="Info">
      <HiQuestionMarkCircle aria-hidden="true" />
    </button>
  );
}