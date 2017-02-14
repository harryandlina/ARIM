package AccurateML.nonLinearRegression

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.optimize.DiffFunction


object ErrorMetric{
   final val RMSE:Int = 1
   final val MAD:Int  = 2
  
}

/**
 * @author Nodalpoint
 * 
 * Abstract definition of an error (or loss) objective function for the case of sum-of-squares.
 * We extend Breeze's DiffFunction so that we can use Breeze optimization algorithms for comparison!
 * 
 * getDim(): Int
 *  Returns the dimensionality of the objective function (the tunable parameters
 *  of the model)
 *  
 * calculate(weights: BDV[Double]): (Double, BDV[Double])
 *  Provided a set of adjustable weights it returns the function value and the gradient vector
 *  
 * hessian(weights: BDV[Double]): BDM[Double]  
 *  Provided a set of adjustable weights it returns the second order derivatives (Hessian matrix)
 *   
 */
trait SumOfSquaresFunction extends  DiffFunction[BDV[Double]] {
  def getDim():Int
  def calculate(weights: BDV[Double]): (Double, BDV[Double])
//  def calculate(weights: BDV[Double],itN:Double): (Double, BDV[Double],Array[Double]) //zf-add
  def calculate(weights: BDV[Double],itN:Int): (Double, BDV[Double],Array[Double],Int)//zf-add
  def hessian(weights: BDV[Double]): BDM[Double]  
 }