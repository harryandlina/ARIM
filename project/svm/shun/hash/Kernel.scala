package AccurateML.svm.shun.hash

/**
  * ICT
  * Created by douyishun on 11/23/16.
  */

import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.collection.Map

/**
  * Define kernel function.
  */
class Kernel extends java.io.Serializable{

  def rbf(gamma: Double)(x_1: Vector, x_2: Vector): Double = {
    math.exp(-1 * gamma * Vectors.sqdist(x_1, x_2))
  }

  def linear(x_1: Vector, x_2: Vector): Double = {
    var sum: Double = 0
    for(i <- 0 until x_1.size) sum += x_1(i) * x_2(i)
    sum
  }

  //Other kernel function...
}

object Kernel{

  /**
    * Compute kernel Matrix.
    * @param indexedPoint Point with index.
    * @param kernelFunc Kernel function.
    * @return Cartesian product, where the calculation is kernel function.
    */
  def computeKernelMatrix(indexedPoint: Map[Long, Vector],
                          kernelFunc: (Vector, Vector) => Double
                         ): Map[(Long, Long), Double] = {

    //define cartesian product
    implicit class Crossable[X](xs: Traversable[X]) {
      def cross[Y](ys: Traversable[Y]) = for { x <- xs; y <- ys } yield (x, y)
    }

    val kernelMatrix = (indexedPoint cross indexedPoint).map(prod => (
      (prod._1._1, prod._2._1), kernelFunc(prod._1._2, prod._2._2)
      ))
    kernelMatrix.toMap
  }

}
