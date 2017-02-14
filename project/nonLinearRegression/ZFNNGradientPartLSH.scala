package AccurateML.nonLinearRegression

import java.io.{File, PrintWriter}

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis
import AccurateML.blas.{ZFBLAS, ZFUtils}

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.io.Source

/**
  * Created by zhangfan on 16/11/17.
  */
class ZFNNGradientPartLSH(
                           fitmodel: NonlinearModel,
                           xydata: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])],
                           r: Double
                           //                           bcWeights:Broadcast[Vector]
                         ) extends Serializable {
  var nnModel: NonlinearModel = fitmodel
  var dim = fitmodel.getDim()
  var data: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])] = xydata
  var trainN: Int = data.count().toInt
  var numFeature: Int = data.first()._1(0).last.features.size
  //data.first()._2.size
  val ratio: Double = r
  var bcWeights: Broadcast[BDV[Double]] = null
  var itN = -1


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
  def zfMapFunc(pit: Iterator[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])]): Iterator[(Vector, Double, Int)] = {
    if (pit.isEmpty) {
      //      val temp=(Vectors.zeros(numFeatures),0.0,0)
      //      List(temp).iterator
      val ans = new ListBuffer[(Vector, Double, Int)]()
      ans.iterator
    } else {
      val jedis = new Jedis("localhost")
      var time = System.currentTimeMillis()

      val partTrain = new ArrayBuffer[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])]()
      while (pit.hasNext)
        partTrain += pit.next()

      val trainN = partTrain.size
      val setN = math.max((trainN * ratio).toInt, 1)
      val chooseLshRound = 1 //set chooseRound
      val weights = bcWeights.value.toArray
      val weightsBDV = new BDV[Double](weights)
      val ans = new ListBuffer[(Vector, Double, Int)]()
      var diffIndexBuffer = new ArrayBuffer[(Int, Double)]()

      //      val zipIndex: Array[Int] = if (ratio == 1) Array.range(0, trainN)
      for (i <- partTrain.indices) {
        val zip = partTrain(i)._1(0).last // lshRound = 1 第一层只有一个zip
        val feat: BDV[Double] = new BDV[Double](zip.features.toArray)
        val per = nnModel.eval(weightsBDV, feat)
        val gper = nnModel.grad(weightsBDV, feat)
        val g1 = 2.0 * (per - zip.label) * gper
        val norm: Double = math.sqrt(g1 dot g1)
        diffIndexBuffer += Tuple2(i, norm)

        //print original data norm vs zip norm
        if ((itN == 1||itN==50||itN==100) && ratio == 1) {
          //only in the first iter and 100% ratio record norms
          val sbnorm = new ArrayBuffer[Double]()
          sbnorm += norm
          sbnorm += partTrain(i)._1.last.size
          for (point <- partTrain(i)._1.last) {
            val pfeat: BDV[Double] = new BDV[Double](point.features.toArray)
            val pper = nnModel.eval(weightsBDV, pfeat)
            val pgper = nnModel.grad(weightsBDV, pfeat)
            val pg1 = 2.0 * (pper - point.label) * pgper
            val pnorm: Double = math.sqrt(pg1 dot pg1)
            sbnorm += pnorm
          }
          jedis.append("norm",itN+","+sbnorm.mkString(",") + "\n")
        }

      }
      val zipIndex = diffIndexBuffer.toArray.sortWith(_._2 > _._2).map(_._1)

      //      for (i <- 0 until trainN) {//使用压缩点
      for (i <- 0 until setN) {
        val zipi = zipIndex(i)
        val chooseRound = {
          if (i < setN) chooseLshRound else 0
        }
        val iter = partTrain(zipi)._1(chooseRound).iterator
        var count: Int = 0
        val gradientSum: Vector = Vectors.zeros(dim)
        var lossSum: Double = 0.0
        while (iter.hasNext) {
          val point = iter.next()
          val feat: BDV[Double] = new BDV[Double](point.features.toArray)
          val per = nnModel.eval(weightsBDV, feat)
          val gper = nnModel.grad(weightsBDV, feat)
          val f1 = 0.5 * Math.pow(point.label - per, 2)
          val g1 = 2.0 * (per - point.label) * gper

          ZFBLAS.axpy(1.0, Vectors.dense(g1.toArray), gradientSum)
          lossSum += f1
          count += 1
        }
        if (itN == 1)
          jedis.append("f", "," + count)
        val tempa = (gradientSum, lossSum, count)
        ans += tempa
      }

      time = System.currentTimeMillis() - time
      jedis.append("lrT", "," + time)
      jedis.append("partN", "," + trainN)
      jedis.append("setN", "," + setN)
      jedis.close()
      ans.iterator
    }
  }

  def calculate(weights: BDV[Double], iN: Int): (BDV[Double], Double, Int) = {
    assert(dim == weights.length)
    itN = iN
    bcWeights = data.context.broadcast(weights)

    val fitModel: NonlinearModel = nnModel
    val n: Int = dim
    val bcDim = data.context.broadcast(dim)
    val mapData = data.mapPartitions(this.zfMapFunc)
    val (gradientSum, lossSum, miniBatchSize) = mapData.reduce((t1, t2) => {
      val temp = Vectors.zeros(t1._1.size)
      ZFBLAS.axpy(1.0, t1._1, temp)
      ZFBLAS.axpy(1.0, t2._1, temp)
      (temp, t1._2 + t2._2, t1._3 + t2._3)
    })
    return (new BDV[Double](gradientSum.toArray), lossSum, miniBatchSize)
  }


}

object ZFNNGradientPartLSH {
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
    val objectPath: String = testPath + args(6)
    val test100: Boolean = args(7).toBoolean
    val weightsPath = args(8)
    val normPath = args(9)



    val dim = new NeuralNetworkModel(numFeature, hiddenNodesN).getDim()
    val w0 = if (initW == -1) {
      val iter = Source.fromFile(new File(weightsPath)).getLines()
      val weights = iter.next().split(",").map(_.toDouble)
      new BDV(weights)
    } else BDV(Array.fill(dim)(initW))

    val mapData: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])] = sc.objectFile(objectPath,1).repartition(1)
    mapData.persist(StorageLevel.MEMORY_AND_DISK)

    val tempn = mapData.map(t => t._1.last.size).sum()

    val test = sc.textFile(testPath).map {
      line =>
        val parts = line.split(',')
        (parts(parts.length - 1).toDouble, Vectors.dense(parts.take(parts.length - 1).map(_.toDouble)))
    }
    println("mapDataPart,"+mapData.getNumPartitions+",testPart,"+test.getNumPartitions+",mapData original pointsN," + tempn)


    val ratioL = if (test100) List(100) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    val vecs = new ArrayBuffer[Vector]()
    val mesb = new ArrayBuffer[Double]()
    for (r <- ratioL) {
      val jedis = new Jedis("localhost")
      jedis.flushAll()
      val ratio = r / 100.0

      val train = mapData
      var trainN = 0.0

      val model: NonlinearModel = new NeuralNetworkModel(numFeature, hiddenNodesN)
      val modelTrain: ZFNNGradientPartLSH = new ZFNNGradientPartLSH(model, train, ratio)
      val hissb = new StringBuilder()
      val w = w0.copy
      for (i <- 1 to itN) {
        val (g1, f1, itTrainN) = modelTrain.calculate(w, i)
        hissb.append("," + f1 / itTrainN)
        val itStepSize = stepSize / itTrainN / math.sqrt(i) //this is stepSize for each iteration
        w -= itStepSize * g1
        trainN += itTrainN
      }
      trainN /= itN
      vecs += Vectors.dense(w.toArray)
      val MSE = test.map { point =>
        val prediction = model.eval(w, new BDV[Double](point._2.toArray))
        (point._1, prediction)
      }.map { case (v, p) => 0.5 * math.pow((v - p), 2) }.mean()

      println()
      val partN = jedis.get("partN").split(",")
      val f = jedis.get("f").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get).sum
      println("all used pointN," + f)
      jedis.close()
      println("partN," + partN.slice(0, math.min(w0.length, 50)).mkString(","))
      println(",ratio," + ratio + ",itN," + itN + ",initW," + w0(0) + ",step," + stepSize + ",hiddenN," + hiddenNodesN + ",trainN," + trainN / 10000.0 + ",testN," + test.count() / 10000.0 + ",numFeature," + numFeature)
      System.out.println("w0= ," + w0.slice(0, math.min(w0.length, 10)).toArray.mkString(","))
      System.out.println("w = ," + w.slice(0, math.min(w.length, 10)).toArray.mkString(","))
      System.out.println(",MSE, " + MSE + ",[" + hissb)
      mesb += MSE
    }

    val n = vecs.length
    val weightEuclidean = new ArrayBuffer[Double]()
    val weightCos = new ArrayBuffer[Double]()

    for (i <- 0 until n) {
      weightEuclidean += ZFUtils.zfEuclideanDistance(vecs(i), vecs.last)
      weightCos += ZFUtils.zfCosin(vecs(i), vecs.last)
    }
    println()
    println(this.getClass.getName + ",step," + stepSize + ",data," + testPath)
    println("ratio,MSE,weightEuclidean,weightCosin")
    for (i <- vecs.toArray.indices) {
      println(ratioL(i) / 100.0 + "," + mesb(i) + "," + weightEuclidean(i) + "," + weightCos(i))
    }
    val writer = new PrintWriter(new File(normPath))//testPath.split(":").last + ".norm"
    val jedis = new Jedis("localhost")
    writer.write(jedis.get("norm"))
    writer.close()
    jedis.close()


  }


}
