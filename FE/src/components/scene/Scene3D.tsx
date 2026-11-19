import {
  useCallback,
  useRef,
  useState,
  type ComponentRef,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { Canvas } from "@react-three/fiber";
import { GizmoHelper, GizmoViewport, OrbitControls } from "@react-three/drei";
import { PointCloud, type PointCloudStats } from "./PointCloud";
import { ZoomControl } from "../ui/ZoomControl";

const MIN_DISTANCE = 12;
const MAX_DISTANCE = 200;
const GIZMO_MARGIN = 88;
const GIZMO_HOVER_RADIUS = 64;
const LEGEND_LEFT = 156;
const LEGEND_WIDTH = 120;
const coordinateFormatter = new Intl.NumberFormat("pl-PL", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

function formatRange(min: number | undefined, max: number | undefined) {
  if (min === undefined || max === undefined) return "[—, —]";
  return `[${coordinateFormatter.format(min)}, ${coordinateFormatter.format(max)}]`;
}

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
  const [cameraPosition, setCameraPosition] = useState<[number, number, number]>([
    0, 0, 50,
  ]);
  const [isGizmoHovered, setIsGizmoHovered] = useState(false);
  const [pointStats, setPointStats] = useState<PointCloudStats | null>(null);

  const handleStatsChange = useCallback((stats: PointCloudStats) => {
    setPointStats(stats);
  }, []);

  const handleControlsChange = () => {
    const controls = controlsRef.current;
    if (controls) {
      setZoom(distanceToZoom(controls.object.position.distanceTo(controls.target)));
      const { x, y, z } = controls.object.position;
      setCameraPosition([x, y, z]);
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

        <PointCloud
          cameraPosition={cameraPosition}
          onStatsChange={handleStatsChange}
        />

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
      <section className="scene-hud" aria-label="Informacje o widocznym kadrze">
        <h2 className="scene-hud-heading">Widoczny kadr</h2>
        <div className="scene-hud-range scene-hud-range-x">
          <span>UMAP 1</span>
          <output>
            {formatRange(pointStats?.visibleBounds?.minX, pointStats?.visibleBounds?.maxX)}
          </output>
        </div>
        <div className="scene-hud-range scene-hud-range-y">
          <span>UMAP 2</span>
          <output>
            {formatRange(pointStats?.visibleBounds?.minY, pointStats?.visibleBounds?.maxY)}
          </output>
        </div>
        <div className="scene-hud-range scene-hud-range-z">
          <span>UMAP 3</span>
          <output>
            {formatRange(pointStats?.visibleBounds?.minZ, pointStats?.visibleBounds?.maxZ)}
          </output>
        </div>
        <div className="scene-hud-count">
          Widoczne: {pointStats?.visibleCount.toLocaleString("pl-PL") ?? "—"} /{" "}
          {pointStats?.totalCount.toLocaleString("pl-PL") ?? "—"} kadrów
        </div>
      </section>
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
