package AccurateML.nonLinearRegression

import java.io.{File, PrintWriter}

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, eigSym}
import breeze.optimize.LineSearch
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis
import AccurateML.blas.ZFUtils

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Sorting

/**
  * Created by zhangfan on 16/11/17.
  */
class ZFNNLineSearchGradientPart(fitmodel: NonlinearModel, xydata: RDD[(Double, Vector)], r: Double) extends SumOfSquaresFunction with Serializable{
  var model: NonlinearModel = fitmodel
  var dim = fitmodel.getDim()
  var data: RDD[(Double, Vector)] = xydata
  var m: Int = data.cache().count().toInt
  var n: Int = data.first()._2.size
  val ratio: Double = r


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
  def calculate(weights: BDV[Double], itN:Int): (Double, BDV[Double], Array[Double], Int) = {
    assert(dim == weights.length)
    val bcW = data.context.broadcast(weights)

    val fitModel: NonlinearModel = model
    val n: Int = dim
    val bcDim = data.context.broadcast(dim)


    val mapData = data.mapPartitions(pit => {
      val ggfs = new ArrayBuffer[(Double, BDV[Double], Double)]()
      while (pit.hasNext) {
        val inc = pit.next()
        val label = inc._1
        val features = inc._2
        val feat: BDV[Double] = new BDV[Double](features.toArray)
        val w: BDV[Double] = new BDV[Double](bcW.value.toArray)
        val per = fitModel.eval(w, feat)
        val gper = fitModel.grad(w, feat)
        val f1 = 0.5 * Math.pow(label - per, 2)
        val g1 = 2.0 * (per - label) * gper

        val t: Double = g1 dot g1
        val gn = math.sqrt(t)
        ggfs += Tuple3(gn, g1, f1)
      }
      val partN = ggfs.length

      val jedis = new Jedis("localhost")
      jedis.append("partN", "," + partN)
      jedis.close()
      val ggf = ggfs.toArray
      Sorting.quickSort(ggf)(Ordering.by[(Double, BDV[Double], Double), Double](-_._1))
      ggf.slice(0, (partN * ratio).toInt).toIterator
    })
    //    val allN = mapData.count()
    val (allGradN, allGrad, allF, allN) = mapData.treeAggregate((new ArrayBuffer[Double], BDV.zeros[Double](n), 0.0, 0))(
      seqOp = (c, v) => (c, v) match {
        case ((gradn, grad, f, n), (agn, ag, af)) =>
          gradn += agn
          (gradn, grad + ag, f + af, n + 1)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case ((gradn1, grad1, f1, n1), (gradn2, grad2, f2, n2)) =>
          val gg = gradn1 ++ gradn2
          grad1 += grad2
          (gg, grad1, f1 + f2, n1 + n2)
      })

    return (allF, allGrad, allGradN.toArray, allN)
  }

  /**
    * for linearSearch API
    *
    * @param weights
    * @return
    */
  def calculate(weights: BDV[Double]): (Double, BDV[Double]) = {
    assert(dim == weights.length)
    val bcW = data.context.broadcast(weights)

    val fitModel: NonlinearModel = model
    val n: Int = dim
    val bcDim = data.context.broadcast(dim)


    val mapData = data.mapPartitions(pit => {
      val ggfs = new ArrayBuffer[(Double, BDV[Double], Double)]()
      while (pit.hasNext) {
        val inc = pit.next()
        val label = inc._1
        val features = inc._2
        val feat: BDV[Double] = new BDV[Double](features.toArray)
        val w: BDV[Double] = new BDV[Double](bcW.value.toArray)
        val per = fitModel.eval(w, feat)
        val gper = fitModel.grad(w, feat)
        val f1 = 0.5 * Math.pow(label - per, 2)
        val g1 = 2.0 * (per - label) * gper

        val t: Double = g1 dot g1
        val gn = math.sqrt(t)
        ggfs += Tuple3(gn, g1, f1)
      }
      val partN = ggfs.length

      val jedis = new Jedis("localhost")
      jedis.append("partN", "," + partN)
      jedis.close()
      val ggf = ggfs.toArray
      Sorting.quickSort(ggf)(Ordering.by[(Double, BDV[Double], Double), Double](-_._1))
      val cancel = ggf.slice(0, 10)
      ggf.slice(0, (partN * ratio).toInt).toIterator
    })
    //    val allN = mapData.count()
    val (allGradN, allGrad, allF, allN) = mapData.treeAggregate((new ArrayBuffer[Double], BDV.zeros[Double](n), 0.0, 0))(
      seqOp = (c, v) => (c, v) match {
        case ((gradn, grad, f, n), (agn, ag, af)) =>
          gradn += agn
          (gradn, grad + ag, f + af, n + 1)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case ((gradn1, grad1, f1, n1), (gradn2, grad2, f2, n2)) =>
          val gg = gradn1 ++ gradn2
          grad1 += grad2
          (gg, grad1, f1 + f2, n1 + n2)
      })

    return (allF, allGrad)
  }


  /**
    * copy from SumOfSquaresFunctionRDD
    */

  def hessian(weights: BDV[Double]): BDM[Double] = {
    System.err.print("MapPartition.hessian\n")
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
    * copy from SumOfSquaresFunctionRDD
    */
  def posDef(H: BDM[Double]): BDM[Double] = {
    System.err.print("MapPartition.posDef\n")
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

object ZFNNLineSearchGradientPart {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)

    val initW: Double = args(0).toDouble
    val stepSize: Double = args(1).toDouble
    val numFeature: Int = args(2).toInt //10
    val hiddenNodesN: Int = args(3).toInt //5
    val itN: Int = args(4).toInt
    val dataPath: String = args(5)
    val test100: Boolean = args(6).toBoolean


    val dataTxt = sc.textFile(dataPath) // "/Users/zhangfan/Documents/nonlinear.f10.n100.h5.txt"
    val dim = new NeuralNetworkModel(numFeature, hiddenNodesN).getDim()
    var w0 = (if (initW == -1) BDV.rand[Double](dim) else BDV(Array.fill(dim)(initW)))
    if(initW== -1){
      val iter = Source.fromFile(new File(dataPath.split(":").last+".nnweights")).getLines()
      val weights=iter.next().split(",").map(_.toDouble)
      w0 = new BDV(weights)
    }

    val ratioL = if (test100) List(100) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

    val vecs = new ArrayBuffer[Vector]()
    val mesb=new ArrayBuffer[Double]()
    for (r <- ratioL) {
      val jedis = new Jedis("localhost")
      jedis.flushAll()
      val ratio = r / 100.0
      val data = dataTxt.map {
        line =>
          val parts = line.split(',')
          (parts(parts.length - 1).toDouble, Vectors.dense(parts.take(parts.length - 1).map(_.toDouble)))
      }
      val train = data.cache()
      var trainN = 0.0

      val model: NonlinearModel = new NeuralNetworkModel(numFeature, hiddenNodesN)
      val modelTrain: ZFNNLineSearchGradientPart = new ZFNNLineSearchGradientPart(model, train, ratio)
      val hissb = new StringBuilder()
      val w = w0.copy
      for (i <- 1 to itN) {
        val (f1, g1, gn, itTrainN) = modelTrain.calculate(w, i)
        hissb.append("," + f1 / itTrainN)
        val dir = -g1
        val fdir = LineSearch.functionFromSearchDirection(modelTrain, w, dir)
        val lsearch = new PolynomialLineSearch(0.0)
        var itStepSize = lsearch.minimize(fdir, stepSize)
        w += itStepSize * dir
        trainN += itTrainN
      }
      trainN /= itN
      vecs += Vectors.dense(w.toArray)
      val MSE = data.map { point =>
        val prediction = model.eval(w, new BDV[Double](point._2.toArray))
        (point._1, prediction)
      }.map { case (v, p) => 0.5 * math.pow((v - p), 2) }.mean()

      val partN = jedis.get("partN").split(",")
      jedis.close()
      println()
      println("partN," + partN.slice(0, math.min(w0.length, 50)).mkString(","))
      println(",ratio," + ratio + ",itN," + itN + ",initW," + w0(0) + ",step," + stepSize + ",hiddenN," + hiddenNodesN + ",trainN," + trainN / 10000.0 + ",testN," + data.count() / 10000.0 + ",numFeatures," + numFeature)
      System.out.println("w0= ," + w0.slice(0, math.min(w0.length, 10)).toArray.mkString(","))
      System.out.println("w = ," + w.slice(0, math.min(w.length, 10)).toArray.mkString(","))
      System.out.println(",MSE, " + MSE + ",[" + hissb)
      mesb += MSE
    }
    val n = vecs.length
    val weightEuclidean = new ArrayBuffer[Double]()
    val weightCos = new ArrayBuffer[Double]()

    for(i<-0 until n){
      weightEuclidean += ZFUtils.zfEuclideanDistance(vecs(i),vecs.last)
      weightCos += ZFUtils.zfCosin(vecs(i),vecs.last)
    }
    println()
    println(this.getClass.getName+",step,"+stepSize+",data,"+dataPath)
    println("ratio,MSE,weightEuclidean,weightCosin")
    for(i<-vecs.toArray.indices){
      println(ratioL(i)/100.0+","+mesb(i)+","+weightEuclidean(i)+","+weightCos(i))
    }


  }


}
