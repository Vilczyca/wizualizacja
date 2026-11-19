import {
  useRef,
  useState,
  type ComponentRef,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { Canvas } from "@react-three/fiber";
import { GizmoHelper, GizmoViewport, OrbitControls } from "@react-three/drei";
import { PointCloud } from "./PointCloud";
import { ZoomControl } from "../ui/ZoomControl";

const MIN_DISTANCE = 12;
const MAX_DISTANCE = 200;
const GIZMO_MARGIN = 88;
const GIZMO_HOVER_RADIUS = 64;
const LEGEND_LEFT = 156;
const LEGEND_WIDTH = 120;

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
  const [isGizmoHovered, setIsGizmoHovered] = useState(false);

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

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const pointerX = event.clientX - bounds.left;
    const pointerY = event.clientY - bounds.top;

    setIsGizmoHovered(
      pointerX >= GIZMO_MARGIN - GIZMO_HOVER_RADIUS &&
        pointerX <= LEGEND_LEFT + LEGEND_WIDTH &&
        pointerY >= bounds.height - GIZMO_MARGIN - GIZMO_HOVER_RADIUS &&
        pointerY <= bounds.height - GIZMO_MARGIN + GIZMO_HOVER_RADIUS,
    );
  };

  return (
    <div
      className="scene-layer"
      onPointerMove={handlePointerMove}
      onPointerLeave={() => setIsGizmoHovered(false)}
    >
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

        <GizmoHelper
          alignment="bottom-left"
          margin={[GIZMO_MARGIN, GIZMO_MARGIN]}
        >
          <GizmoViewport
            scale={50}
            axisColors={["#a85b63", "#6f9b78", "#6686ad"]}
            axisScale={[0.8, 0.045, 0.045]}
            labels={["1", "2", "3"]}
            labelColor="#ffffff"
            font="24px system-ui, sans-serif"
            axisHeadScale={1}
          />
        </GizmoHelper>
      </Canvas>
      <div
        className={`orientation-legend${isGizmoHovered ? " orientation-legend-visible" : ""}`}
        aria-label="Legenda osi"
        aria-hidden={!isGizmoHovered}
      >
        <span className="orientation-legend-item orientation-legend-item-x">
          UMAP 1
        </span>
        <span className="orientation-legend-item orientation-legend-item-y">
          UMAP 2
        </span>
        <span className="orientation-legend-item orientation-legend-item-z">
          UMAP 3
        </span>
      </div>
      <ZoomControl value={zoom} onChange={handleZoomChange} />
    </div>
  );
}
