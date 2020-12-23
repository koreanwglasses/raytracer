import * as tf from "@tensorflow/tfjs";
import { lerp } from "./math";
import { TensorWithShape, withShape as ensureShape } from "./tensor-types";

export class Rays<N extends number = number> {
  constructor(
    public origins: TensorWithShape<[N, 3]>,
    public directions: TensorWithShape<[N, 3]>
  ) {}

  get numRays(): N {
    return this.origins.shape[0];
  }
}

export interface Geometry {
  intersect<N extends number>(
    rays: Rays<N>,
    opts?: { reverseSide: boolean }
  ): { hitDepths: TensorWithShape<[N]> };
}

export interface Material {}

export interface Occluder {
  geometry: Geometry;
  material: Material;
}

export function closestHits<N extends number>(
  occluders: Occluder[],
  rays: Rays<N>
) {
  const N = rays.numRays;

  let closestHitIndices = tf.ones([N]).neg() as TensorWithShape<[N]>;
  let closestHitDepths = tf.fill([N], Infinity) as TensorWithShape<[N]>;

  occluders.forEach((occluder, index) => {
    const { hitDepths } = occluder.geometry.intersect(rays);
    const mask = ensureShape(hitDepths.less(closestHitDepths), [N]);

    closestHitIndices = ensureShape(lerp(closestHitIndices, index, mask), [N]);
    closestHitDepths = ensureShape(closestHitDepths.minimum(hitDepths), [N]);
  });

  return { closestHitIndices, closestHitDepths };
}
