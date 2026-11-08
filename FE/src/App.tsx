import { lazy, Suspense, useState } from "react";
import { UploadButton } from "./components/ui/UploadButton";
import { UploadModal } from "./components/ui/UploadModal";
import { InfoButton } from "./components/ui/InfoButton";
import { InfoModal } from "./components/ui/InfoModal";
import "./App.css";

const Scene3D = lazy(() =>
  import("./components/scene/Scene3D").then(({ Scene3D }) => ({ default: Scene3D })),
);

function App() {
  const [activeModal, setActiveModal] = useState<"upload" | "info" | null>(null);

  return (
    <div className="app-shell">
      <div className="app-toolbar">
        <UploadButton onClick={() => setActiveModal("upload")} />
        <InfoButton onClick={() => setActiveModal("info")} />
      </div>
      <UploadModal
        isOpen={activeModal === "upload"}
        onClose={() => setActiveModal(null)}
      />
      <InfoModal
        isOpen={activeModal === "info"}
        onClose={() => setActiveModal(null)}
      />

      <Suspense fallback={null}>
        <Scene3D />
      </Suspense>
    </div>
  );
}

export default App;
