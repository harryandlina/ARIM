package AccurateML.nonLinearRegression

import java.io.File

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis
import AccurateML.blas.ZFUtils

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.io.Source

/**
  * Created by zhangfan on 16/11/17.
  */

class ZFNNLSHPart3Statistic(
                             fitmodel: NonlinearModel,
                             data: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])],
                             r: Double
                           ) extends Serializable {
  var nnModel: NonlinearModel = fitmodel
  var dim = fitmodel.getDim()
  var train: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])] = data
  var trainN: Int = train.count().toInt
  var numFeature: Int = train.first()._1(0)(0).features.size
  val nnRatio: Double = r
  var bcWeights: Broadcast[BDV[Double]] = null
  var nnItN = -1


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
  def zfNNMap(pit: Iterator[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])]): Iterator[(BDV[Double], Double, Int)] = {
    if (pit.isEmpty) {
      val ans = new ListBuffer[(BDV[Double], Double, Int)]()
      ans.iterator
    } else {
      val vsRatio = nnRatio
      var nnMapT = System.currentTimeMillis()
      val jedis = new Jedis("localhost")
      val objectData = new ArrayBuffer[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])]()
      while (pit.hasNext) {
        objectData += pit.next()
      }
      val zipN = objectData.size
      val weights = bcWeights.value.toArray
      val weightsBDV = new BDV[Double](weights)
      val ans = new ArrayBuffer[(BDV[Double], Double, Int)]()


      val zipIndex = {
        var diffIndexBuffer = new ArrayBuffer[(Int, Double, Double)]()
        //        var diffIndexBuffer = new ArrayBuffer[(Int, Double)]()
        for (i <- objectData.indices) {
          val zip = objectData(i)._1(0).last // lshRound = 1 第一层只有一个zip
          val feat: BDV[Double] = new BDV[Double](zip.features.toArray)
          val per = nnModel.eval(weightsBDV, feat)
          val gper = nnModel.grad(weightsBDV, feat)
          val g1 = 2.0 * (per - zip.label) * gper
          val norm: Double = math.sqrt(g1 dot g1)
          //          diffIndexBuffer += Tuple3(i,gper.toArray.map(math.abs).sum ,math.abs(per - zip.label))
          diffIndexBuffer += Tuple3(i, norm, 0.0)
          //          diffIndexBuffer += Tuple2(i, math.abs(per - zip.label))
        }
        //        diffIndexBuffer.toArray.sortBy(t=>(t._2<=1E-2,-math.abs(t._3))).map(_._1) //from big to small
        diffIndexBuffer.toArray.sortWith(_._2 > _._2).map(_._1)
      }

      val vsZipN:Int = math.max(zipN * vsRatio, 1).toInt
      var vsPointN = 0
      val vsZipsG1 = new ArrayBuffer[Double]()
      val vsZipOnlyG1 = new ArrayBuffer[Double]()

      val vsZipsDiff = new ArrayBuffer[Double]()
      val vsZipOnlyDiff = new ArrayBuffer[Double]()


      //      val cancelsb = new ArrayBuffer[String]()

      for (i <- 0 until vsZipN) {
        val zipi = zipIndex(i)
        val zip: LabeledPoint = objectData(zipi)._1(0).last
        val zfeat: BDV[Double] = new BDV[Double](zip.features.toArray)
        val zper = nnModel.eval(weightsBDV, zfeat)
        val zgper = nnModel.grad(weightsBDV, zfeat)
        val zg1 = 2.0 * (zper - zip.label) * zgper
        val znorm: Double = math.sqrt(zg1 dot zg1)

        {
          //          cancelsb += "\n@znorm,diff,gperAbsSum,ziplabel,zipFeaAbsSum"
          //          cancelsb += ""+znorm+","+math.abs(zper - zip.label)+","+zgper.toArray.map(math.abs).sum+","+zip.label+","+zip.features.toArray.map(math.abs).sum

        }

        vsZipOnlyG1 += znorm
        vsZipOnlyDiff += math.abs(zper - zip.label)

        val chooseRound = 1
        val iter = objectData(zipi)._1(chooseRound).iterator
        while (iter.hasNext) {
          val point = iter.next()
          val feat: BDV[Double] = new BDV[Double](point.features.toArray)
          val per = nnModel.eval(weightsBDV, feat)
          val gper = nnModel.grad(weightsBDV, feat)
          val g1 = 2.0 * (per - point.label) * gper
          val norm: Double = math.sqrt(g1 dot g1)
          //          cancelsb += ""+norm+","+(per - point.label)+","+gper.toArray.map(math.abs).sum+","+point.label+","+point.features.toArray.map(math.abs).sum
          vsZipsG1 += norm
          vsZipsDiff += math.abs(per - point.label)
          vsPointN += 1
        }
      }
      //      val writer = new java.io.PrintWriter(new java.io.File("cancel"))
      //      writer.write(cancelsb.mkString("\n"))
      //      writer.close()

      var vsPoints = new ArrayBuffer[Tuple3[Double, Double, Double]]()
      //      var vsPointsG1 = new ArrayBuffer[Double]()
      //      var vsPointsDiff = new ArrayBuffer[Double]()
      var allCount = 0
      var cancel=0.0
      for (zipi <- 0 until zipN) {
        val iter = objectData(zipi)._1(1).iterator
        while (iter.hasNext) {
          val point = iter.next()
          val feat: BDV[Double] = new BDV[Double](point.features.toArray)
          val per = nnModel.eval(weightsBDV, feat)
          val gper = nnModel.grad(weightsBDV, feat)
          val g1 = 2.0 * (per - point.label) * gper
          val norm: Double = math.sqrt(g1 dot g1)
          vsPoints += Tuple3(norm, gper.toArray.map(math.abs).sum, math.abs(per - point.label))
          allCount += 1

          cancel += point.label
        }

      }
      jedis.append("cancel",","+cancel)

      //      vsPoints =vsPoints.sortBy(t=>(t._2<=1E-2,-math.abs(t._3))).slice(0,vsPointN)
      vsPoints = vsPoints.sortWith(_._1 > _._1).slice(0, vsPointN)
      //      vsPointsG1 = vsPointsG1.sortWith(_ > _).slice(0, vsPointN)
      //      val vsPointsG1Min = vsPointsG1.toArray.min

      var shootN = 0
      //      for (i <- 0 until vsPointN) {
      //        if (vsZipsG1(i) >= vsPointsG1Min)
      //          shootN += 1
      //      }
      //      val t = new BDV[Double](Array[Double](vsZipN, allCount, vsZipOnlyG1.sum, vsZipsG1.sum, vsPoints.map(_._2).sum,
      //        vsZipOnlyDiff.sum, vsZipsDiff.sum, vsPoints.map(_._3).sum))



      val t = new BDV[Double](Array[Double](vsZipN, allCount, vsZipOnlyG1.sum, vsZipsG1.sum, vsPoints.map(_._1).sum,
        vsZipOnlyDiff.sum, vsZipsDiff.sum, vsPoints.map(_._3).sum))
      ans += Tuple3(t, shootN, vsPointN)
      ans.toArray.iterator
    }
  }

  def calculate(weights: BDV[Double], iN: Int): (BDV[Double], Double, Int) = {
    assert(dim == weights.length)
    nnItN = iN
    bcWeights = train.context.broadcast(weights)

    val fitModel: NonlinearModel = nnModel
    val n: Int = dim
    val bcDim = train.context.broadcast(dim)
    val mapData = train.mapPartitions(this.zfNNMap)
    val (gradientSum, lossSum, miniBatchSize) = mapData.treeAggregate(BDV.zeros[Double](8), 0.0, 0)(
      seqOp = (c, v) => (c, v) match {
        case ((grad, f, n), (ag, af, an)) =>
          (grad + ag, f + af, n + an)
      },
      combOp = (u1, u2) => (u1, u2) match {
        case ((grad1, f1, n1), (grad2, f2, n2)) =>
          (grad1 + grad2, f1 + f2, n1 + n2)
      }
    )
    return (gradientSum, lossSum, miniBatchSize)
  }


}

object ZFNNLSHPart3Statistic {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)

    val initW: Double = args(0).toDouble
    val stepSize: Double = args(1).toDouble
    val numFeature: Int = args(2).toInt //10
    val hiddenNodesN: Int = args(3).toInt //5
    val nnItN: Int = args(4).toInt
    val testPath: String = args(5)
    val dataPath: String = args(6)
    val test100: Boolean = args(7).toBoolean
    val weightsPath = args(8)
    val normPath = args(9)

    val itqbitN = args(10).toInt
    val itqitN = args(11).toInt
    val itqratioN = args(12).toInt //from 1 not 0
    val minPartN = args(13).toInt
    val upBound = args(14).toInt
    val splitN = args(15).toDouble


    val dim = new NeuralNetworkModel(numFeature, hiddenNodesN).getDim()
    val w0 = if (initW == -1) {
      val iter = Source.fromFile(new File(weightsPath)).getLines()
      val weights = iter.next().split(",").map(_.toDouble)
      new BDV(weights)
    } else BDV(Array.fill(dim)(initW))


    val data = sc.textFile(dataPath, minPartN).map(line => {
      val vs = line.split(",").map(_.toDouble)
      val features = vs.slice(0, vs.size - 1)
      LabeledPoint(vs.last, Vectors.dense(features))
    })

    val jedis = new Jedis("localhost")
    jedis.flushAll()
    val oHash = new ZFHash3(itqbitN, itqitN, itqratioN, upBound, splitN)
    val objectData = data.mapPartitions(oHash.zfHashMap).persist(StorageLevel.MEMORY_AND_DISK)

    val test = sc.textFile(testPath).map {
      line =>
        val parts = line.split(',')
        (parts(parts.length - 1).toDouble, Vectors.dense(parts.take(parts.length - 1).map(_.toDouble)))
    }
    val on = objectData.count()
    println()
    println("dataPart," + data.getNumPartitions + ",objectDataPart," + objectData.getNumPartitions + ",testPart," + test.getNumPartitions)

    val readT = jedis.get("readT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val hashIterT = jedis.get("hashIterT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val zipT = jedis.get("zipT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val HashMapT = jedis.get("HashMapT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val clusterN = jedis.get("clusterN").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)

    val itIncreSVDT = jedis.get("itIncreSVDT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itToBitT = jedis.get("itToBitT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
    val itOthersT = jedis.get("itOthersT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)



    println("objectDataN," + on + ",itqbitN," + itqbitN + ",itqitN," + itqitN ++ ",itqratioN," + itqratioN + ",splitN," + splitN)
    println("readT," + readT.sum + ",hashIterT," + hashIterT.sum + ",zipT," + zipT.sum + ",HashMapT," + HashMapT.sum + ",clusterN," + clusterN.size + ",/," + clusterN.sum + ",AVG," + clusterN.sum / clusterN.size + ",[," + clusterN.slice(0, 20).mkString(",") + ",],")

    println("itIncreSVDT," + itIncreSVDT.sum + ",itToBitT," + itToBitT.sum + ",itOthersT," + itOthersT.sum)
    jedis.close()


//    val ratioL = if (test100) List(100) else List(1, 2, 3, 4, 5)
    val ratioL = if (test100) List(100) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

    val pointN = new ArrayBuffer[Double]()
    val zipOnlySumG = new ArrayBuffer[Double]()
    val zipSumG = new ArrayBuffer[Double]()
    val pointSumG = new ArrayBuffer[Double]()
    val stN = new ArrayBuffer[Double]()
    val zipN = new ArrayBuffer[Double]()

    val zipOnlySumDiff = new ArrayBuffer[Double]()
    val zipSumDiff = new ArrayBuffer[Double]()
    val pointSumDiff = new ArrayBuffer[Double]()

    for (r <- ratioL) {




      val nnRatio = r / 100.0
      val train = objectData
      val model: NonlinearModel = new NeuralNetworkModel(numFeature, hiddenNodesN)
      val modelTrain: ZFNNLSHPart3Statistic = new ZFNNLSHPart3Statistic(model, train, nnRatio)
      val w = w0.copy

      for (i <- 1 to 1) {
        val (g1, shootN, pointsN) = modelTrain.calculate(w, i)
        //zipN, allCount,vsZipOnlyG1.toArray.sum, vsZipsSum, vsPointsSum
        //vsZipOnlyDiff.sum,vsZipsDiff.sum,vsPoints.map(_._3).sum


        val jedis = new Jedis("localhost")
        val cancel = jedis.get("cancel").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get).sorted
        println(r+",\t"+cancel.mkString(","))
        jedis.flushAll()
        jedis.close()

        val izipN = g1(0)
        val allPointsN = g1(1)
        val vszipOnlySum = g1(2)
        val vsZipsSum = g1(3) // /pointsN
        val vsPointsSum = g1(4)

        val vszipOnlyDiffSum = g1(5)
        val vsZipsDiffSum = g1(6) // /pointsN
        val vsPointsDiffSum = g1(7)


        pointN += pointsN
        zipOnlySumG += vszipOnlySum
        zipSumG += vsZipsSum
        pointSumG += vsPointsSum
        stN += shootN
        zipN += izipN

        zipOnlySumDiff += vszipOnlyDiffSum
        zipSumDiff += vsZipsDiffSum
        pointSumDiff += vsPointsDiffSum
      }

    }
    var i = pointN.size - 1
    while (i > 0) {
      pointN(i) -= pointN(i - 1)
      zipSumG(i) -= zipSumG(i - 1)
      pointSumG(i) -= pointSumG(i - 1)
      //      stN(i) -= stN(i - 1)
      zipN(i) -= zipN(i - 1)
      zipOnlySumG(i) -= zipOnlySumG(i - 1)

      zipOnlySumDiff(i) -= zipOnlySumDiff(i - 1)
      zipSumDiff(i) -= zipSumDiff(i - 1)
      pointSumDiff(i) -= pointSumDiff(i - 1)

      i -= 1
    }
    for (i <- 0 until ratioL.size) {
      val r = ratioL(i)
      if (pointN(i) > 0)
        println("%" + r + ",shootN," + stN(i) + ",pointN," + pointN(i) + ",zipOnlyG," + zipOnlySumG(i) / zipN(i) + ",zipG," + zipSumG(i) / pointN(i) + ",pointG," + pointSumG(i) / pointN(i)
          + ",zipOnlyDiff," + zipOnlySumDiff(i) / zipN(i) + ",zipDiff," + zipSumDiff(i) / pointN(i) + ",pointDiff," + pointSumDiff(i) / pointN(i))
      else
        println("%" + r + "NO DATA")
    }

  }


}
