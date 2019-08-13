package AccurateML.lsh

import breeze.linalg._
import breeze.linalg.svd.SVD
import breeze.stats.mean

import scala.util.Random._

object PCA1 {

  def zfpca(x:DenseMatrix[Double],dimension: Int): DenseMatrix[Double] ={
    val alltime = System.currentTimeMillis()
    var time=System.currentTimeMillis()
    val covmat = cov(x)
    val SVD(_,_,loadings) = svd(covmat)
    println("svd,"+(System.currentTimeMillis()-time))

     time=System.currentTimeMillis()
    // val center = mean(x,Axis._0).toDenseVector
    val center = mean(x,Axis._0).inner
    println("center,"+(System.currentTimeMillis()-time))

    time=System.currentTimeMillis()
    //    val partLoadings:DenseMatrix[Double] = loadings(0 until dimension,::)
    /**
      * Translate the original data points to the PC axes.
      */
//     val scores:DenseMatrix[Double] =  (partLoadings * x.t).t
     val scores:DenseMatrix[Double] =  (loadings * (x(*,::) - center).t).t
    println("score,"+(System.currentTimeMillis()-time))
    println("alltime,"+(System.currentTimeMillis()-alltime))
    scores
  }
  def main(args: Array[String]) {
    val data=DenseMatrix.rand[Double](100,1000)
//      val data = DenseMatrix(
//        (2.0, 4.5, 5.1),
//        (2.0, 2.5, -5.1),
//        (-8.0, -3.0, 6.4),
//        (8.0, 0.5, -6.5),
//        (-4.0, -4.5, 0.1))
    val nbits = 2
    val pca = breeze.linalg.princomp(data)
//    val v = pca.scores

    val zfpcav=zfpca(data,nbits)
    println()


  }

}