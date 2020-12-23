import * as tf from "@tensorflow/tfjs";
import { Tensor, TensorLike } from "@tensorflow/tfjs";

export function lerp(
  a: Tensor | TensorLike,
  b: Tensor | TensorLike,
  t: Tensor | TensorLike
) {
  return tf.mul(a, tf.sub(1, t)).add(tf.mul(b, t));
}
