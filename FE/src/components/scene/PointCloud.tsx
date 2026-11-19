import { useMemo } from "react";
import { Generator, inBox } from "maath/random";

const DUMMY_COUNT = 10000;

export function PointCloud() {
  const positions = useMemo(() => {
    return inBox(
      new Float32Array(DUMMY_COUNT * 3),
      { sides: 100, center: [0, 0, 0] },
      new Generator(123456789),
    ) as Float32Array;
  }, []);

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial color="#ffffff" size={0.4} sizeAttenuation />
    </points>
  );
}