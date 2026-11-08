import { lazy, Suspense, useState } from "react";
import "./App.css";

const Scene3D = lazy(() =>
  import("./components/scene/Scene3D").then(({ Scene3D }) => ({ default: Scene3D })),
);

function App() {

  return (
    <div className="app-shell">
      <Suspense fallback={null}>
        <Scene3D />
      </Suspense>
    </div>
  );
}

export default App;
