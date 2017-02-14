package AccurateML.nonLinearRegression

import java.io.File

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis
import AccurateML.blas.ZFUtils

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by zhangfan on 16/11/17.
  */
class ZFNNPart(fitmodel: NonlinearModel, xydata: RDD[LabeledPoint]) extends Serializable {
  var model: NonlinearModel = fitmodel
  var dim = fitmodel.getDim()
  var data: RDD[LabeledPoint] = xydata
  var m: Int = data.cache().count().toInt
  var n: Int = data.first().features.size


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
  def calculate(weights: BDV[Double], itN: Int): (Double, BDV[Double], Array[Double], Int) = {
    assert(dim == weights.length)
    val bcW = data.context.broadcast(weights)

    val fitModel: NonlinearModel = model
    val n: Int = dim
    val bcDim = data.context.broadcast(dim)


    val mapData: RDD[(Double, BDV[Double], Double)] = data.mapPartitions(pit => {
      val jedis = new Jedis("localhost")
      val nnMapT = System.currentTimeMillis()
      val ggfs = new ArrayBuffer[(Double, BDV[Double], Double)]()
      while (pit.hasNext) {
        val inc = pit.next()
        val label = inc.label
        val features = inc.features
        val feat: BDV[Double] = new BDV[Double](features.toArray)
        val w: BDV[Double] = new BDV[Double](bcW.value.toArray)
        val per = fitModel.eval(w, feat)
        val gper = fitModel.grad(w, feat)
        val f1 = 0.5 * Math.pow(label - per, 2)
        val g1 = 2.0 * (per - label) * gper

        //        val t: Double = g1 dot g1
        //        val gn = math.sqrt(t)
        //        ggfs += Tuple3(gn, g1, f1)
        ggfs += Tuple3(0, g1, f1)
      }
      var ggf = ggfs.toArray
      val partN = ggf.size

      jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
      jedis.append("partN", "," + partN)
      jedis.close()

      //      Sorting.quickSort(ggf)(Ordering.by[(Double, BDV[Double], Double), Double](-_._1))
      //      ggf.slice(0, (partN * ratio).toInt).toIterator
      ggf.toIterator
    }).persist(StorageLevel.MEMORY_AND_DISK_2)
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


}

object ZFNNPart {
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
    val testPath: String = args(5)
    val dataPath: String = args(6)
    val test100: Boolean = args(7).toBoolean
    val weightsPath = args(8)
    val isSparse = args(9).toBoolean
    val minPartN = args(10).toInt

    //    val ratioL = if (test100) List(100) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    val ratioL = if (test100) List(100) else List(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100)

    val vecs = new ArrayBuffer[Vector]()
    val mesb = new ArrayBuffer[Double]()
    val nntimesb = new ArrayBuffer[Double]()

    for (r <- ratioL) {
      val dataTxt = sc.textFile(dataPath, minPartN) // "/Users/zhangfan/Documents/nonlinear.f10.n100.h5.txt"
      val dim = new NeuralNetworkModel(numFeature, hiddenNodesN).getDim()
      val w0 = if (initW == -1) {
        val iter = Source.fromFile(new File(weightsPath)).getLines()
        val weights = iter.next().split(",").map(_.toDouble)
        new BDV(weights)
      } else BDV(Array.fill(dim)(initW))
      val jedis = new Jedis("localhost")
      jedis.flushAll()
      val ratio = r / 100.0
      val data = if (!isSparse) {
        dataTxt.map(line => {
          val vs = line.split(",").map(_.toDouble)
          val features = vs.slice(0, vs.size - 1)
          LabeledPoint(vs.last, Vectors.dense(features))
        })
      } else {
        MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
      }

      val splits = data.randomSplit(Array(0.8, 0.2), seed = System.currentTimeMillis())
      val train = if (testPath.size > 3) data.sample(false, ratio).cache() else splits(0).sample(false, ratio).cache()
      val test = if (testPath.size > 3) {
        println("testPath,\t" + testPath)
        //        MLUtils.loadLibSVMFile(sc, testPath)
        if (!isSparse) {
          sc.textFile(testPath).map(line => {
            val vs = line.split(",").map(_.toDouble)
            val features = vs.slice(0, vs.size - 1)
            LabeledPoint(vs.last, Vectors.dense(features))
          })
        } else {
          MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN)
        }
      } else splits(1)



      //      val train = data.sample(false, ratio).cache()
      var trainN = 0.0
      val model: NonlinearModel = new NeuralNetworkModel(numFeature, hiddenNodesN)
      val modelTrain: ZFNNPart = new ZFNNPart(model, train)
      val hissb = new StringBuilder()
      val w = w0.copy
      for (i <- 1 to itN) {
        val (f1, g1, gn, itTrainN) = modelTrain.calculate(w, i)
        hissb.append("," + f1 / itTrainN)
        val itStepSize = stepSize / itTrainN / math.sqrt(i) //this is stepSize for each iteration
        w -= itStepSize * g1
        trainN += itTrainN

      }
      trainN /= itN
      //      val test = if (!isSparse) {
      //        sc.textFile(testPath).map(line => {
      //          val vs = line.split(",").map(_.toDouble)
      //          val features = vs.slice(0, vs.size - 1)
      //          LabeledPoint(vs.last, Vectors.dense(features))
      //        })
      //      } else {
      //        MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN)
      //      }


      val MSE = test.map { point =>
        val prediction = model.eval(w, new BDV[Double](point.features.toArray))
        (point.label, prediction)
      }.map { case (v, p) => 0.5 * math.pow((v - p), 2) }.mean()

      val partN = jedis.get("partN").split(",")
      val nnMapT = jedis.get("nnMapT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)

      jedis.close()
      println()

      println("dataPart," + data.getNumPartitions + ",trainPart," + train.getNumPartitions + ",testPart," + test.getNumPartitions)
      println("partN," + partN.slice(0, math.min(partN.length, 20)).mkString(","))
      println(",ratio," + ratio + ",itN," + itN + ",initW," + w0(0) + ",step," + stepSize + ",hiddenN," + hiddenNodesN + ",trainN," + trainN / 10000.0 + ",testN," + test.count() / 10000.0 + ",numFeatures," + numFeature)
      System.out.println("w0= ," + w0.slice(0, math.min(w0.length, 10)).toArray.mkString(","))
      System.out.println("w = ," + w.slice(0, math.min(w.length, 10)).toArray.mkString(","))
      System.out.println(",MSE, " + MSE + ",[" + hissb)
      println("nnMapT," + nnMapT.sum)

      vecs += Vectors.dense(w.toArray)
      mesb += MSE
      nntimesb += nnMapT.sum
    }
    val n = vecs.length

    println()
    println(this.getClass.getName + ",step," + stepSize + ",data," + dataPath)
    println("ratio,MSE,nnMapT")
    for (i <- vecs.toArray.indices) {
      println(ratioL(i) / 100.0 + "," + mesb(i) + "," + nntimesb(i))
    }


  }


}
