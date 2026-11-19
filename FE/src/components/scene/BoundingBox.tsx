import { useMemo } from "react";
import { Billboard, Text } from "@react-three/drei";

export interface PointCloudBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
}

type Axis = "x" | "y" | "z";
type Vector3Tuple = [number, number, number];

interface BoundingBoxProps {
  bounds: PointCloudBounds;
  cameraPosition: Vector3Tuple;
}

interface RearEdge {
  axis: Axis;
  start: Vector3Tuple;
  end: Vector3Tuple;
  normal: Vector3Tuple;
  min: number;
  max: number;
}

interface Tick {
  axis: Axis;
  start: Vector3Tuple;
  end: Vector3Tuple;
  labelPosition: Vector3Tuple;
  label: string;
}

const AXIS_COLORS: Record<Axis, string> = {
  x: "#a85b63",
  y: "#6f9b78",
  z: "#6686ad",
};

function getNiceStep(min: number, max: number, targetCount = 5) {
  const range = max - min;
  if (range <= 0) return 1;

  const rawStep = range / (targetCount - 1);
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const fraction = rawStep / magnitude;
  const niceFraction =
    fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 2.5 ? 2.5 : fraction <= 5 ? 5 : 10;

  return niceFraction * magnitude;
}

function getTickValues(min: number, max: number) {
  const step = getNiceStep(min, max);
  const epsilon = step * 1e-8;
  const first = Math.ceil((min - epsilon) / step) * step;
  const values: number[] = [];

  for (let value = first; value <= max + epsilon; value += step) {
    values.push(Number(value.toPrecision(6)));
  }

  return values;
}

function formatTickValue(value: number) {
  return Number(value.toPrecision(4)).toString();
}

function createRearEdges(
  bounds: PointCloudBounds,
  cameraSides: [boolean, boolean, boolean],
): RearEdge[] {
  const center: Vector3Tuple = [
    (bounds.minX + bounds.maxX) / 2,
    (bounds.minY + bounds.maxY) / 2,
    (bounds.minZ + bounds.maxZ) / 2,
  ];
  const min: Vector3Tuple = [bounds.minX, bounds.minY, bounds.minZ];
  const max: Vector3Tuple = [bounds.maxX, bounds.maxY, bounds.maxZ];
  const far: Vector3Tuple = min.map((_, axis) =>
    cameraSides[axis] ? min[axis] : max[axis],
  ) as Vector3Tuple;
  const outward: Vector3Tuple = far.map((value, axis) =>
    value >= center[axis] ? 1 : -1,
  ) as Vector3Tuple;
  const diagonal = Math.SQRT1_2;

  return [
    {
      axis: "x",
      start: [bounds.minX, far[1], far[2]],
      end: [bounds.maxX, far[1], far[2]],
      normal: [0, outward[1] * diagonal, outward[2] * diagonal],
      min: bounds.minX,
      max: bounds.maxX,
    },
    {
      axis: "y",
      start: [far[0], bounds.minY, far[2]],
      end: [far[0], bounds.maxY, far[2]],
      normal: [outward[0] * diagonal, 0, outward[2] * diagonal],
      min: bounds.minY,
      max: bounds.maxY,
    },
    {
      axis: "z",
      start: [far[0], far[1], bounds.minZ],
      end: [far[0], far[1], bounds.maxZ],
      normal: [outward[0] * diagonal, outward[1] * diagonal, 0],
      min: bounds.minZ,
      max: bounds.maxZ,
    },
  ];
}

function createBoxEdgePositions(bounds: PointCloudBounds) {
  const min: Vector3Tuple = [bounds.minX, bounds.minY, bounds.minZ];
  const max: Vector3Tuple = [bounds.maxX, bounds.maxY, bounds.maxZ];
  const positions: number[] = [];

  for (let axis = 0; axis < 3; axis++) {
    const fixedAxes = [0, 1, 2].filter((candidate) => candidate !== axis);

    for (const firstSide of [0, 1]) {
      for (const secondSide of [0, 1]) {
        const start = [...min] as Vector3Tuple;
        const end = [...min] as Vector3Tuple;
        start[fixedAxes[0]] = firstSide ? max[fixedAxes[0]] : min[fixedAxes[0]];
        end[fixedAxes[0]] = start[fixedAxes[0]];
        start[fixedAxes[1]] = secondSide ? max[fixedAxes[1]] : min[fixedAxes[1]];
        end[fixedAxes[1]] = start[fixedAxes[1]];
        end[axis] = max[axis];
        positions.push(...start, ...end);
      }
    }
  }

  return new Float32Array(positions);
}

export function BoundingBox({ bounds, cameraPosition }: BoundingBoxProps) {
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  const centerZ = (bounds.minZ + bounds.maxZ) / 2;
  const cameraOnPositiveX = cameraPosition[0] >= centerX;
  const cameraOnPositiveY = cameraPosition[1] >= centerY;
  const cameraOnPositiveZ = cameraPosition[2] >= centerZ;
  const { boxEdges, rearEdges, ticks, tickPositions, maxDimension, titlePositions } =
    useMemo(() => {
      const boxEdges = createBoxEdgePositions(bounds);
      const rearEdges = createRearEdges(bounds, [
        cameraOnPositiveX,
        cameraOnPositiveY,
        cameraOnPositiveZ,
      ] as [boolean, boolean, boolean]);
      const maxDimension = Math.max(
        bounds.maxX - bounds.minX,
        bounds.maxY - bounds.minY,
        bounds.maxZ - bounds.minZ,
        1,
      );
      const tickSize = maxDimension * 0.018;
      const labelOffset = maxDimension * 0.045;
      const titleOffset = maxDimension * 0.07;
      const ticks = rearEdges.flatMap((edge) =>
        getTickValues(edge.min, edge.max).map((value): Tick => {
          const progress = (value - edge.min) / (edge.max - edge.min || 1);
          const position: Vector3Tuple = [
            edge.start[0] + (edge.end[0] - edge.start[0]) * progress,
            edge.start[1] + (edge.end[1] - edge.start[1]) * progress,
            edge.start[2] + (edge.end[2] - edge.start[2]) * progress,
          ];
          const end: Vector3Tuple = [
            position[0] + edge.normal[0] * tickSize,
            position[1] + edge.normal[1] * tickSize,
            position[2] + edge.normal[2] * tickSize,
          ];
          const labelPosition: Vector3Tuple = [
            position[0] + edge.normal[0] * labelOffset,
            position[1] + edge.normal[1] * labelOffset,
            position[2] + edge.normal[2] * labelOffset,
          ];

          return {
            axis: edge.axis,
            start: position,
            end,
            labelPosition,
            label: formatTickValue(value),
          };
        }),
      );
      const tickPositions = new Float32Array(
        ticks.flatMap((tick) => [...tick.start, ...tick.end]),
      );
      const titlePositions = rearEdges.map((edge) => ({
        axis: edge.axis,
        position: [
          edge.end[0] +
            (edge.end[0] - edge.start[0]) * 0.08 +
            edge.normal[0] * titleOffset,
          edge.end[1] +
            (edge.end[1] - edge.start[1]) * 0.08 +
            edge.normal[1] * titleOffset,
          edge.end[2] +
            (edge.end[2] - edge.start[2]) * 0.08 +
            edge.normal[2] * titleOffset,
        ] as Vector3Tuple,
      }));

      return {
        boxEdges,
        rearEdges,
        ticks,
        tickPositions,
        maxDimension,
        titlePositions,
      };
    }, [bounds, cameraOnPositiveX, cameraOnPositiveY, cameraOnPositiveZ]);

  return (
    <group>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[boxEdges, 3]} />
        </bufferGeometry>
        <lineBasicMaterial
          color="#9aa8bb"
          transparent
          opacity={0.22}
          depthWrite={false}
        />
      </lineSegments>

      <lineSegments>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[tickPositions, 3]} />
        </bufferGeometry>
        <lineBasicMaterial
          color="#b4c0d0"
          transparent
          opacity={0.65}
          depthWrite={false}
        />
      </lineSegments>

      {ticks.map((tick, index) => (
        <Billboard key={`${tick.axis}-${index}`} position={tick.labelPosition}>
          <Text
            fontSize={maxDimension * 0.022}
            color="#d6deea"
            outlineWidth={maxDimension * 0.0015}
            outlineColor="#101522"
            depthOffset={0}
          >
            {tick.label}
          </Text>
        </Billboard>
      ))}

      {titlePositions.map(({ axis, position }) => (
        <Billboard key={axis} position={position}>
          <Text
            fontSize={maxDimension * 0.03}
            color={AXIS_COLORS[axis]}
            outlineWidth={maxDimension * 0.0015}
            outlineColor="#101522"
            depthOffset={0}
          >
            {`UMAP ${axis === "x" ? "1" : axis === "y" ? "2" : "3"}`}
          </Text>
        </Billboard>
      ))}

      {rearEdges.map((edge) => (
        <line key={`rear-${edge.axis}`}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              args={[new Float32Array([...edge.start, ...edge.end]), 3]}
            />
          </bufferGeometry>
          <lineBasicMaterial
            color={AXIS_COLORS[edge.axis]}
            transparent
            opacity={0.55}
            depthWrite={false}
          />
        </line>
      ))}
    </group>
  );
}
