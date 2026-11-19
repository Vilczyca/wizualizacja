import { useMemo } from "react";
import { Box3, BufferAttribute } from "three";
import { Generator, inBox } from "maath/random";
import { BoundingBox, type PointCloudBounds } from "./BoundingBox";

const DUMMY_COUNT = 10000;

interface PointCloudProps {
  cameraPosition: [number, number, number];
}

export function PointCloud({ cameraPosition }: PointCloudProps) {
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