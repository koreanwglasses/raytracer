import * as tf from "@tensorflow/tfjs";

export type TensorWithShape<Shape extends number[]> = tf.Tensor & {
  shape: Shape;
};

export function tensorHasShape<Shape extends number[]>(
  tensor: tf.Tensor,
  shape: Shape
): tensor is TensorWithShape<Shape> {
  return tensor.shape
    .map((value, i) => shape[i] === null || value === shape[i])
    .reduce((a, b) => a && b, true);
}

export function withShape<Shape extends number[]>(
  tensor: tf.Tensor,
  shape: [...Shape]
): TensorWithShape<Shape> {
  if (tensorHasShape(tensor, shape)) {
    return tensor;
  } else {
    throw new Error(
      `shape assertion failed: expected ${shape}, got ${tensor.shape}`
    );
  }
}
