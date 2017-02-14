package AccurateML.nonLinearRegression


import java.io.File

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.optimize.LineSearch

import scala.io.Source

//import breeze.optimize.LineSearch
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}
import AccurateML.blas.ZFUtils

import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangfan on 16/11/11.
  */
object ZFNNLineSearch {

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
      val ratio = r / 100.0
      val data = dataTxt.map {
        line =>
          val parts = line.split(',')
          (parts(parts.length - 1).toDouble, Vectors.dense(parts.take(parts.length - 1).map(_.toDouble)))
      }
      val train = data.sample(false, ratio).cache()
      val trainN = train.count()

      val model: NonlinearModel = new NeuralNetworkModel(numFeature, hiddenNodesN)
      val modelTrain: SumOfSquaresFunction = new SumOfSquaresFunctionRDD(model, train)
      val hissb = new StringBuilder()
      var w = w0.copy
      for (i <- 1 to itN) {
//        val (f1, g1) = modelTrain.calculate(w)
//        hissb.append("," + f1/trainN)
//        val itStepSize = stepSize / trainN / math.sqrt(i) //this is stepSize for each iteration
//        w -= itStepSize * g1
        val (f1, g1) = modelTrain.calculate(w)
        hissb.append("," + f1 / trainN)
        // Create descent direction (-g1)
        var dir = -g1
        val fdir = LineSearch.functionFromSearchDirection(modelTrain, w, dir)
        val lsearch = new PolynomialLineSearch(0.0)
        //  1.0 setpSize is the starting value for itStepSize. You can use a larger number here
        var itStepSize = lsearch.minimize(fdir, stepSize) // val itStepSize = stepSize / trainN / math.sqrt(i)

        w = w + itStepSize * dir //w -= itStepSize * g1
      }
      vecs += Vectors.dense(w.toArray)
      val MSE = data.map { point =>
        val prediction = model.eval(w, new BDV[Double](point._2.toArray))
        (point._1, prediction)
      }.map { case (v, p) => 0.5 * math.pow((v - p), 2) }.mean()

      println()
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
