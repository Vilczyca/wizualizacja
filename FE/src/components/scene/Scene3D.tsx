import { useRef, useState, type ComponentRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { PointCloud } from "./PointCloud";
import { ZoomControl } from "../ui/ZoomControl";

const MIN_DISTANCE = 12;
const MAX_DISTANCE = 200;

function distanceToZoom(distance: number) {
  return (
    (100 * Math.log(MAX_DISTANCE / distance)) /
    Math.log(MAX_DISTANCE / MIN_DISTANCE)
  );
}

function zoomToDistance(zoom: number) {
  return MAX_DISTANCE * (MIN_DISTANCE / MAX_DISTANCE) ** (zoom / 100);
}

export function Scene3D() {
  const controlsRef = useRef<ComponentRef<typeof OrbitControls>>(null);
  const [zoom, setZoom] = useState(distanceToZoom(50));

  const handleControlsChange = () => {
    const controls = controlsRef.current;
    if (controls) {
      setZoom(distanceToZoom(controls.object.position.distanceTo(controls.target)));
    }
  };

  const handleZoomChange = (value: number) => {
    setZoom(value);

    const controls = controlsRef.current;
    if (!controls) return;

    const direction = controls.object.position
      .clone()
      .sub(controls.target)
      .normalize();
    controls.object.position
      .copy(controls.target)
      .addScaledVector(direction, zoomToDistance(value));
    controls.update();
  };

  return (
    <div className="scene-layer">
      <Canvas
        camera={{ position: [0, 0, 50], fov: 60 }}
        dpr={[1, 1.5]}
        frameloop="demand"
      >
        <OrbitControls
          ref={controlsRef}
          makeDefault
          minDistance={MIN_DISTANCE}
          maxDistance={MAX_DISTANCE}
          onChange={handleControlsChange}
        />

        <PointCloud />
      </Canvas>
      <ZoomControl value={zoom} onChange={handleZoomChange} />
    </div>
  );
}
