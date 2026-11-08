import { useMemo } from "react";

const DUMMY_COUNT = 10000;

export function PointCloud() {
  const positions = useMemo(() => {
    const values = new Float32Array(DUMMY_COUNT * 3);
    let seed = 123456789;

    for (let i = 0; i < values.length; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      values[i] = (seed / 0x100000000 - 0.5) * 100;
    }

    return values;
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