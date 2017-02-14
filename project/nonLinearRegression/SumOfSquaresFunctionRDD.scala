package AccurateML.nonLinearRegression

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector, Vectors}
import org.apache.spark.rdd._
import AccurateML.blas.ZFBLAS

import scala.collection.mutable.ArrayBuffer


/**
  * @author Nodalpoint
  *         Implementation of the sum-of-squares objective function using Spark RDDs
  *
  *         Properties
  *         model: NonlinearModel -> The nonlinear model that defines the function
  *         data: RDD[(Double, Vector)] -> The training data used to calculate the loss in a form of a Spark RDD
  *         that contains target, input pairs.
  *         [(t1, x1), (t2, x2), ..., (tm, xm)]
  */
class SumOfSquaresFunctionRDD(fitmodel: NonlinearModel, xydata: RDD[(Double, Vector)]) extends SumOfSquaresFunction {
  var model: NonlinearModel = fitmodel
  var dim = fitmodel.getDim()
  var data: RDD[(Double, Vector)] = xydata
  var m: Int = data.cache().count().toInt
  var n: Int = data.first()._2.size


  /**
    * Return the objective function dimensionality which is essentially the model's dimensionality
    */
  def getDim(): Int = {
    return this.dim
  }

  /**
    * This method is inherited by Breeze DiffFunction. Given an input vector of weights it returns the
    * objective function and the first order derivative.
    * It operates using treeAggregate action on the training pair data.
    * It is essentially the same implementation as the one used for the Stochastic Gradient Descent
    * Partial subderivative vectors are calculated in the map step
    * val per = fitModel.eval(w, feat)
    * val gper = fitModel.grad(w, feat)
    * and are aggregated by summation in the reduce part.
    */
  def calculate(weights: BDV[Double]): (Double, BDV[Double]) = {
    assert(dim == weights.length)
    val bcW = data.context.broadcast(weights)

    val fitModel: NonlinearModel = model
    val n: Int = dim
    val bcDim = data.context.broadcast(dim)
    val (grad, f) = data.treeAggregate((Vectors.zeros(n), 0.0))(
      seqOp = (c, v) => (c, v) match {
        case ((grad, loss), (label, features)) =>
          //fitModel.setWeights(bcW.value)
          val feat: BDV[Double] = new BDV[Double](features.toArray)
          val w: BDV[Double] = new BDV[Double](bcW.value.toArray)
          val per = fitModel.eval(w, feat)
          val gper = fitModel.grad(w, feat)
          var f1 = 0.5 * Math.pow(label - per, 2)
          var g1 = 2.0 * (per - label) * gper

          val gradBDV = new BDV[Double](grad.toArray)
          var newgrad = Vectors.dense((g1 + gradBDV).toArray)

          (newgrad, loss + f1)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case ((grad1, loss1), (grad2, loss2)) =>
          //axpy(1.0, grad2, grad1)
          val grad1BDV = new BDV[Double](grad1.toArray)
          val grad2BDV = new BDV[Double](grad2.toArray)
          var newgrad = Vectors.dense((grad1BDV + grad2BDV).toArray)
          (newgrad, loss1 + loss2)
      })




    val gradBDV = new BDV[Double](grad.toArray)
    return (f, gradBDV)
  }

  def calculate(weights: BDV[Double], itN: Int): (Double, BDV[Double],Array[Double],Int) = {
    System.err.println("SumOfSquaresFunctionRDD.calculate(w,i)")
    assert(dim == weights.length)
    val bcW = data.context.broadcast(weights)

    val fitModel: NonlinearModel = model
    val n: Int = dim
    val bcDim = data.context.broadcast(dim)
    val (grad, f,gn) = data.treeAggregate((Vectors.zeros(n), 0.0, new ArrayBuffer[Double]))(
      seqOp = (c, v) => (c, v) match {
        case ((grad, loss,gn), (label, features)) =>
          //fitModel.setWeights(bcW.value)
          val feat: BDV[Double] = new BDV[Double](features.toArray)
          val w: BDV[Double] = new BDV[Double](bcW.value.toArray)
          val per = fitModel.eval(w, feat)
          val gper = fitModel.grad(w, feat)
          var f1 = 0.5 * Math.pow(label - per, 2)
          var g1 = 2.0 * (per - label) * gper

          val gradBDV = new BDV[Double](grad.toArray)
          var newgrad = Vectors.dense((g1 + gradBDV).toArray)

          val adot=ZFBLAS.dot(Vectors.dense(g1.toArray),Vectors.dense(g1.toArray))
          gn += math.sqrt(adot)
          (newgrad, loss + f1,gn)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case ((grad1, loss1,gn1), (grad2, loss2,gn2)) =>
          //axpy(1.0, grad2, grad1)
          val grad1BDV = new BDV[Double](grad1.toArray)
          val grad2BDV = new BDV[Double](grad2.toArray)
          var newgrad = Vectors.dense((grad1BDV + grad2BDV).toArray)
          (newgrad, loss1 + loss2,gn1++gn2)
      })




    val gradBDV = new BDV[Double](grad.toArray)
    return (f, gradBDV,gn.toArray,0)
  }
//  def calculate(weights: BDV[Double], itN: Double): (Double, BDV[Double]) = {
//    assert(dim == weights.length)
//    val bcW = data.context.broadcast(weights)
//
//    val fitModel: NonlinearModel = model
//    val n: Int = dim
//    val bcDim = data.context.broadcast(dim)
//    val (grad, f) = data.treeAggregate((Vectors.zeros(n), 0.0))(
//      seqOp = (c, v) => (c, v) match {
//        case ((grad, loss), (label, features)) =>
//          //fitModel.setWeights(bcW.value)
//          val feat: BDV[Double] = new BDV[Double](features.toArray)
//          val w: BDV[Double] = new BDV[Double](bcW.value.toArray)
//          val per = fitModel.eval(w, feat)
//          val gper = fitModel.grad(w, feat)
//          var f1 = 0.5 * Math.pow(label - per, 2)
//          var g1 = 2.0 * (per - label) * gper
//
//          if(itN == -1){
//            val nodes = 1 //only print 1 hidden node
//            val n = 9
//            val x = feat
//
//            val ss = new ArrayBuffer[Double]()
//            val wx = new ArrayBuffer[Double]()
//            for (i <- 0 to nodes - 1) {
//              var arg: Double = 0
//              for (j <- 0 to n - 1) {
//                arg = arg + x(j) * w(i * (n + 2) + j)
//              }
//              arg = arg + w(i * (n + 2) + n)
//              var sig: Double = 1.0 / (1.0 + Math.exp(-arg))
//
//              gper(i * (n + 2) + n + 1) = sig
//              gper(i * (n + 2) + n) = w(i * (n + 2) + n + 1) * sig * (1 - sig)
//              for (j <- 0 to n - 1) {
//                gper(i * (n + 2) + j) = x(j) * w(i * (n + 2) + n + 1) * sig * (1 - sig)
//                ss += sig * (1 - sig)
//                wx += x(j) * w(i * (n + 2) + n + 1)
//              }
//              println(itN + ",sig*(1-sig)," + ss.mkString(","))
//              println(itN + ",wx," + wx.mkString(","))
//              println(itN + ",diff," + (per - label))
//              println(itN + ",g1," + g1.toArray.mkString(","))
//            }
//          }
//
//          val gradBDV = new BDV[Double](grad.toArray)
//          var newgrad = Vectors.dense((g1 + gradBDV).toArray)
//
//          (newgrad, loss + f1)
//      },
//      combOp = (c1, c2) => (c1, c2) match {
//        case ((grad1, loss1), (grad2, loss2)) =>
//          //axpy(1.0, grad2, grad1)
//          val grad1BDV = new BDV[Double](grad1.toArray)
//          val grad2BDV = new BDV[Double](grad2.toArray)
//          var newgrad = Vectors.dense((grad1BDV + grad2BDV).toArray)
//          (newgrad, loss1 + loss2)
//      })
//
//
//
//
//    val gradBDV = new BDV[Double](grad.toArray)
//    return (f, gradBDV)
//  }

  /**
    * This method is caclulates the Hessian matrix approximation using Jacobian matrix and
    * the algorithm of Wilamowsky and Yu.
    * It operates using treeAggregate action on the training pair data.
    * Partial subhessian matrices are calculated in the map step
    * val gper = fitModel.grad(w, feat)
    * val hper : BDM[Double] = (gper * gper.t)
    * and aggregated by summation in the reduce part.
    * Extra care is taken to transform between Breee and Spark DenseVectors.
    */
  def hessian(weights: BDV[Double]): BDM[Double] = {
    val bcW = data.context.broadcast(weights)

    val fitModel: NonlinearModel = model
    val n: Int = dim
    val bcDim = data.context.broadcast(dim)

    val (hess) = data.treeAggregate((new DenseMatrix(n, n, new Array[Double](n * n))))(
      seqOp = (c, v) => (c, v) match {
        case ((hess), (label, features)) =>
          val w: BDV[Double] = new BDV[Double](bcW.value.toArray)
          val feat: BDV[Double] = new BDV[Double](features.toArray)
          val gper = fitModel.grad(w, feat)
          val hper: BDM[Double] = (gper * gper.t)
          val hperDM: DenseMatrix = new DenseMatrix(n, n, hper.toArray)

          for (i <- 0 until n * n) {
            hess.values(i) = hperDM.values(i) + hess.values(i)
          }

          (hess)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case ((hess1), (hess2)) =>
          for (i <- 0 until n * n) {
            hess1.values(i) = hess1.values(i) + hess2.values(i)
          }
          (hess1)
      })

    var hessBDM: BDM[Double] = new BDM[Double](n, n, hess.toArray)
    var Hpos = posDef(hessBDM)

    return Hpos
  }


  /**
    * Helper method that uses eigenvalue decomposition to enforce positive definiteness of the Hessian
    */
  def posDef(H: BDM[Double]): BDM[Double] = {
    var n = H.rows
    var m = H.cols
    var dim = model.getDim()
    var Hpos: BDM[Double] = BDM.zeros(n, m)

    var eigens = eigSym(H)
    var oni: BDV[Double] = BDV.ones[Double](dim)
    var diag = eigens.eigenvalues
    var vectors = eigens.eigenvectors


    //diag = diag :+ (1.0e-4)
    for (i <- 0 until dim) {
      if (diag(i) < 1.0e-4) {
        diag(i) = 1.0e-4
      }

    }


    var I = BDM.eye[Double](dim)
    for (i <- 0 until dim) {
      I(i, i) = diag(i)
    }

    Hpos = vectors * I * (vectors.t)
    return Hpos
  }
}