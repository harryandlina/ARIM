package AccurateML.nonLinearRegression

import breeze.linalg.{DenseVector => BDV}

/**
 * @author Nodalpoint
 * All models used in least squares fitting must implement methods from this
 * abstract class
 */

trait NonlinearModel {
  /*
   * Model's output for a single input instance t
   * phi(w; t) for fixed set of weights w
   */
  def eval(w: BDV[Double], x: BDV[Double]): Double
   /*
   * Model's derivative for a single input instance t
   * d(phi(w; t)) / dw
   */
  def grad(w: BDV[Double], x: BDV[Double]): BDV[Double]
  /*
   * Model's dimensionality 
   */
  def getDim(): Int
  /*
   * Model's derivative for a single input instance t
   * d(phi(w; t)) / dw calculated numerically using forward differences
   */
  def gradnumer(w: BDV[Double], x: BDV[Double]): BDV[Double]
}