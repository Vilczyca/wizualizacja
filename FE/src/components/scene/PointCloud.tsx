import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Box3, BufferAttribute, Matrix4 } from "three";
import { Generator, inBox } from "maath/random";
import { BoundingBox, type PointCloudBounds } from "./BoundingBox";

const DUMMY_COUNT = 10000;
const STATS_UPDATE_INTERVAL = 0.1;

export interface PointCloudStats {
  visibleCount: number;
  totalCount: number;
  visibleBounds: PointCloudBounds | null;
}

interface PointCloudProps {
  cameraPosition: [number, number, number];
  onStatsChange: (stats: PointCloudStats) => void;
}

function getVisiblePointStats(
  positions: Float32Array,
  viewProjection: Matrix4,
): PointCloudStats {
  const matrix = viewProjection.elements;
  let visibleCount = 0;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;

  for (let index = 0; index < positions.length; index += 3) {
    const x = positions[index];
    const y = positions[index + 1];
    const z = positions[index + 2];
    const clipX = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    const clipY = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    const clipZ = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
    const clipW = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];

    if (
      clipW <= 0 ||
      clipX < -clipW ||
      clipX > clipW ||
      clipY < -clipW ||
      clipY > clipW ||
      clipZ < -clipW ||
      clipZ > clipW
    ) {
      continue;
    }

    visibleCount++;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    minZ = Math.min(minZ, z);
    maxZ = Math.max(maxZ, z);
  }

  return {
    visibleCount,
    totalCount: positions.length / 3,
    visibleBounds:
      visibleCount === 0
        ? null
        : { minX, maxX, minY, maxY, minZ, maxZ },
  };
}

export function PointCloud({ cameraPosition, onStatsChange }: PointCloudProps) {
  const { positions, bounds } = useMemo(() => {
    const positions = inBox(
      new Float32Array(DUMMY_COUNT * 3),
      { sides: 100, center: [0, 0, 0] },
      new Generator(123456789),
    ) as Float32Array;

    const box = new Box3().setFromBufferAttribute(new BufferAttribute(positions, 3));

    return {
      positions,
      bounds: {
        minX: box.min.x,
        maxX: box.max.x,
        minY: box.min.y,
        maxY: box.max.y,
        minZ: box.min.z,
        maxZ: box.max.z,
      } satisfies PointCloudBounds,
    };
  }, []);
  const viewProjection = useMemo(() => new Matrix4(), []);
  const lastViewProjection = useRef(new Matrix4());
  const hasScanned = useRef(false);
  const lastScanTime = useRef(-Infinity);

  useFrame(({ camera, clock }) => {
    if (clock.elapsedTime - lastScanTime.current < STATS_UPDATE_INTERVAL) return;
    lastScanTime.current = clock.elapsedTime;

    camera.updateMatrixWorld();
    viewProjection.multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse,
    );

    if (hasScanned.current && viewProjection.equals(lastViewProjection.current)) {
      return;
    }

    lastViewProjection.current.copy(viewProjection);
    hasScanned.current = true;
    onStatsChange(getVisiblePointStats(positions, viewProjection));
  });

  return (
    <group>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        </bufferGeometry>
        <pointsMaterial color="#ffffff" size={0.4} sizeAttenuation />
      </points>
      <BoundingBox bounds={bounds} cameraPosition={cameraPosition} />
    </group>
  );
}