import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { PointCloud } from "./PointCloud";

export function Scene3D() {
  return (
    <div className="scene-layer">
      <Canvas
        camera={{ position: [0, 0, 50], fov: 60 }}
        dpr={[1, 1.5]}
        frameloop="demand"
      >
        <OrbitControls makeDefault />

        <PointCloud />
      </Canvas>
    </div>
  );
}
