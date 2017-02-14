package AccurateML.nonLinearRegression

import java.io.File

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis
import AccurateML.blas.{ZFBLAS, ZFUtils}
import AccurateML.lsh.IncreSVD

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.io.Source

/**
  * Created by zhangfan on 16/11/17.
  */
class ZFHash3(
               itqbitN: Int,
               itqitN: Int,
               itqratioN: Int,
               upBound: Int,
               spN: Double,
               sparse: Boolean = false
               //                           bcWeights:Broadcast[Vector]
             ) extends Serializable {

  val itqBitN = itqbitN
  val itqItN = itqitN
  val itqRatioN = itqratioN
  val upperBound = upBound
  val splitN = spN
  val featureVarianceIndex = new ArrayBuffer[Int]()
  val isSparse = sparse

  def directFeatureHashIterative(data: ArrayBuffer[LabeledPoint], indexs: Array[Int], hashIt: Int, bigN: String): mutable.HashMap[String, ArrayBuffer[Int]] = {

    var time = System.currentTimeMillis()
    val jedis = new Jedis("localhost")
    val aMap = new mutable.HashMap[String, ArrayBuffer[Int]]()
    var numFeatures = data.last.features.size
    var numInc = indexs.size

    //    val zfsvd = new IncreSVD(data, indexs, itqBitN, itqItN, itqRatioN)
    //    zfsvd.calcFeaatures(hashIt)
    //    val v: BDM[Double] = zfsvd.userFeas.t

    if (hashIt == 0) {
      val featureVariance = new Array[Double](numFeatures)
      for (f <- 0 until numFeatures) {
        val feas = new ArrayBuffer[Double]() // new Array[Double](numInc)
        var i = 0
        while (i < numInc) {
          feas += data(i).features(f)
          i += itqRatioN
        }
        val μ = ZFUtils.zfmean(feas.toArray)
        val σ = ZFUtils.zfstddev(feas.toArray, μ)
        featureVariance(f) = σ
      }
      featureVarianceIndex ++= featureVariance.zipWithIndex.sortWith(_._1 > _._1).map(_._2)
    }
    val feaArray = new Array[Double](indexs.size)
    val feai = featureVarianceIndex(hashIt % numFeatures) //hashIt%numFeatures
    for (i <- 0 until indexs.size)
      feaArray(i) = data(indexs(i)).features(feai)
    val v: BDM[Double] = new BDM[Double](indexs.size, itqBitN, feaArray)
    jedis.append("itIncreSVDT", "," + (System.currentTimeMillis() - time))


    time = System.currentTimeMillis()
    var u = BDM.zeros[Int](v.rows, v.cols)
    val sliceN: Double = v.rows / splitN.toDouble
    for (f <- 0 until itqBitN) {
      val tempa = v(::, f).toArray
      //      val meadf = ZFUtils.zfFindMeadian(tempa)
      val sortedIndex: Array[Int] = (tempa zip Array.range(0, tempa.size)).sortWith(_._1 < _._1).unzip._2
      for (i <- 0 until u.rows) {
        u(i, f) = (sortedIndex(i) / sliceN).toInt
      }
    }
    jedis.append("itToBitT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val prefix: String = bigN + "@"
    for (i <- 0 until u.rows) {
      val key = u(i, ::).inner.toArray.mkString(",")
      val mapKey = key + prefix
      val aset = aMap.getOrElse(mapKey, new ArrayBuffer[Int]())
      aset += indexs(i)
      aMap.update(mapKey, aset)
    }
    var bN: Int = 0
    val ans = new mutable.HashMap[String, ArrayBuffer[Int]]()
    jedis.append("itOthersT", "," + (System.currentTimeMillis() - time))

    for (temp <- aMap.toList.sortWith(_._1 < _._1)) {
      if (temp._2.size <= upperBound || hashIt >= 50) {
        ans += temp
      } else {
        ans ++= directFeatureHashIterative(data, temp._2.toArray, hashIt + 1, bigN + bN)
        bN += 1
      }
    }
    jedis.close()
    ans
  }


  def hashIterative(data: ArrayBuffer[LabeledPoint], indexs: Array[Int], hashIt: Int, bigN: String): mutable.HashMap[String, ArrayBuffer[Int]] = {

    var time = System.currentTimeMillis()
    val jedis = new Jedis("localhost")
    val aMap = new mutable.HashMap[String, ArrayBuffer[Int]]()
    var numFeatures = data.last.features.size
    var numInc = indexs.size

    val zfsvd = new IncreSVD(data, indexs, itqBitN, itqItN, itqRatioN)
    zfsvd.calcFeaatures(hashIt, isSparse)
    val v: BDM[Double] = zfsvd.userFeas.t
    jedis.append("itIncreSVDT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    var u = BDM.zeros[Int](v.rows, v.cols)
    val sliceN: Double = v.rows / splitN.toDouble
    for (f <- 0 until itqBitN) {
      val tempa = v(::, f).toArray
      //      val meadf = ZFUtils.zfFindMeadian(tempa)
      val sortedIndex: Array[Int] = (tempa zip Array.range(0, tempa.size)).sortWith(_._1 < _._1).unzip._2
      for (i <- 0 until u.rows) {
        u(i, f) = (sortedIndex(i) / sliceN).toInt
      }
    }
    jedis.append("itToBitT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val prefix: String = bigN + "@"
    for (i <- 0 until u.rows) {
      val key = u(i, ::).inner.toArray.mkString(",")
      val mapKey = key + prefix
      val aset = aMap.getOrElse(mapKey, new ArrayBuffer[Int]())
      aset += indexs(i)
      aMap.update(mapKey, aset)
    }
    var bN: Int = 0
    val ans = new mutable.HashMap[String, ArrayBuffer[Int]]()
    jedis.append("itOthersT", "," + (System.currentTimeMillis() - time))

    for (temp <- aMap.toList.sortWith(_._1 < _._1)) {
      if (temp._2.size <= upperBound || hashIt >= 50) {
        ans += temp
      } else {
        ans ++= hashIterative(data, temp._2.toArray, hashIt + 1, bigN + bN)
        bN += 1
      }
    }
    jedis.close()
    ans
  }

  def zfHashMapChoice(pit: Iterator[LabeledPoint], isSparse: Boolean): Iterator[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])] = {
    if (!isSparse) {
      zfHashMap(pit)
    } else {
      zfHashMapSparse(pit)
    }
  }

  def zfHashMapSparse(pit: Iterator[LabeledPoint]): Iterator[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])] = {
    val HashMapT = System.currentTimeMillis()
    val jedis = new Jedis("localhost")

    var time = System.currentTimeMillis()
    var partData = new ArrayBuffer[LabeledPoint]()
    while (pit.hasNext) {
      partData += pit.next()
    }
    jedis.append("readT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val numFeatures = partData.last.features.size
    val indexs = Array.range(0, partData.size)
    //    val aMap = directFeatureHashIterative(partData, indexs, 0, "")
    val aMap = hashIterative(partData, indexs, 0, "")
    jedis.append("hashIterT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val ansAll = new ArrayBuffer[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])]()
    val mapit = aMap.iterator
    while (mapit.hasNext) {
      val (key, value) = mapit.next()
      val points = ArrayBuffer[LabeledPoint]()
      val zipfea = Vectors.zeros(numFeatures)
      var ziplabel = 0.0
      for (i <- value) {
        val point = partData(i)
        points += point
        ZFBLAS.axpy(1.0, point.features, zipfea)
        ziplabel += point.label
      }
      val divn = 1.0 / value.size
      ZFBLAS.scal(divn, zipfea)
      ziplabel *= divn
      val zip = new LabeledPoint(ziplabel, zipfea)
      val zips = new Array[ArrayBuffer[LabeledPoint]](2)
      zips(0) = ArrayBuffer(zip)
      zips(1) = points
      jedis.append("clusterN", "," + points.size)
      ansAll += Tuple3(zips, Array[ArrayBuffer[(Int, Int)]](), new ArrayBuffer[mutable.HashSet[Int]])
    }
    jedis.append("zipT", "," + (System.currentTimeMillis() - time))
    jedis.append("HashMapT", "," + (System.currentTimeMillis() - HashMapT))
    jedis.close()
    ansAll.iterator
  }


  def zfHashMap(pit: Iterator[LabeledPoint]): Iterator[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])] = {
    val HashMapT = System.currentTimeMillis()
    val jedis = new Jedis("localhost")

    var time = System.currentTimeMillis()
    var partData = new ArrayBuffer[LabeledPoint]()
    while (pit.hasNext) {
      partData += pit.next()
    }
    jedis.append("readT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val numFeatures = partData.last.features.size
    val indexs = Array.range(0, partData.size)
    //    val aMap = directFeatureHashIterative(partData, indexs, 0, "")
    val aMap = hashIterative(partData, indexs, 0, "")
    jedis.append("hashIterT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val ansAll = new ArrayBuffer[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])]()
    val mapit = aMap.iterator
    while (mapit.hasNext) {
      val (key, value) = mapit.next()
      val points = ArrayBuffer[LabeledPoint]()
      val zipfea = Vectors.zeros(numFeatures)
      var ziplabel = 0.0
      for (i <- value) {
        val point = partData(i)
        points += point
        ZFBLAS.axpy(1.0, point.features, zipfea)
        ziplabel += point.label
      }
      val divn = 1.0 / value.size
      ZFBLAS.scal(divn, zipfea)
      ziplabel *= divn
      val zip = if (!isSparse) new LabeledPoint(ziplabel, zipfea) else new LabeledPoint(ziplabel, zipfea.toSparse)
      val zips = new Array[ArrayBuffer[LabeledPoint]](2)
      zips(0) = ArrayBuffer(zip)
      zips(1) = points
      jedis.append("clusterN", "," + points.size)
      ansAll += Tuple3(zips, Array[ArrayBuffer[(Int, Int)]](), new ArrayBuffer[mutable.HashSet[Int]])
    }
    jedis.append("zipT", "," + (System.currentTimeMillis() - time))
    jedis.append("HashMapT", "," + (System.currentTimeMillis() - HashMapT))
    jedis.close()
    ansAll.iterator
  }


}

class ZFNNLSHPart3(
                    fitmodel: NonlinearModel,
                    data: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])],
                    r: Double
                  ) extends Serializable {
  var nnModel: NonlinearModel = fitmodel
  var dim = fitmodel.getDim()
  var train: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])] = data
  //  var trainN: Int = train.count().toInt
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
      var nnMapT = System.currentTimeMillis()
      val jedis = new Jedis("localhost")
      val objectData = new ArrayBuffer[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])]()
      while (pit.hasNext) {
        objectData += pit.next()
      }
      val zipN = objectData.size
      val setN: Int = math.max((zipN * nnRatio).toInt, 1)
      val chooseLshRound = 1 //set chooseRound
      val weights = bcWeights.value.toArray
      val weightsBDV = new BDV[Double](weights)
      val ans = new ArrayBuffer[(BDV[Double], Double, Int)]()

      val zipIndex = if (nnRatio == 2) Array.range(0, zipN) // nnRatio == 1,if(nnRatio==2)相当于if(false)
      else {
        var diffIndexBuffer = new ArrayBuffer[(Int, Double, Double)]()
        for (i <- objectData.indices) {
          val zip = objectData(i)._1(0).last // lshRound = 1 第一层只有一个zip
          val feat: BDV[Double] = new BDV[Double](zip.features.toArray)
          val per = nnModel.eval(weightsBDV, feat)
          val gper = nnModel.grad(weightsBDV, feat)
          val g1 = 2.0 * (per - zip.label) * gper
          val norm: Double = math.sqrt(g1 dot g1)
          //          diffIndexBuffer += Tuple3(i, gper.toArray.map(math.abs).sum, math.abs(per - zip.label))
          diffIndexBuffer += Tuple3(i, norm, 0.0)
          //          diffIndexBuffer += Tuple2(i, math.abs(per - zip.label))

        }
        //        diffIndexBuffer.toArray.sortBy(t => (t._2 <= 1E-2, -math.abs(t._3))).map(_._1) //from big to small

        diffIndexBuffer.toArray.sortWith(_._2 > _._2).map(_._1)
      }

      //      for (i <- 0 until zipN) {//使用压缩点
      for (i <- 0 until setN) {
        val zipi = zipIndex(i)

        {
          val zip = objectData(zipi)._1(0).last // lshRound = 1 第一层只有一个zip
        val feat: BDV[Double] = new BDV[Double](zip.features.toArray)
          val per = nnModel.eval(weightsBDV, feat)
          val gper = nnModel.grad(weightsBDV, feat)
          val g1 = 2.0 * (per - zip.label) * gper
          val norm: Double = math.sqrt(g1 dot g1)
        }

        val chooseRound = {
          if (i < setN) chooseLshRound else 0
        }
        val iter = objectData(zipi)._1(chooseRound).iterator
        var count: Int = 0
        while (iter.hasNext) {
          val point = iter.next()
          val feat: BDV[Double] = new BDV[Double](point.features.toArray)
          val per = nnModel.eval(weightsBDV, feat)
          val gper = nnModel.grad(weightsBDV, feat)
          val f1 = 0.5 * Math.pow(point.label - per, 2)
          val g1 = 2.0 * (per - point.label) * gper


          ans += Tuple3(g1, f1, 1)
          count += 1
        }
      }


      jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
      jedis.append("zipN", "," + zipN)
      jedis.append("setN", "," + setN)
      jedis.close()
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
    val (gradientSum, lossSum, miniBatchSize) = mapData.treeAggregate(BDV.zeros[Double](n), 0.0, 0)(
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

object ZFNNLSHPart3 {
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
    val isSparse = args(9).toBoolean

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


    val data = if (!isSparse) {
      sc.textFile(dataPath, minPartN).map(line => {
        val vs = line.split(",").map(_.toDouble)
        val features = vs.slice(0, vs.size - 1)
        LabeledPoint(vs.last, Vectors.dense(features))
      })
    } else {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
    }

    val splits = data.randomSplit(Array(0.8, 0.2), seed = System.currentTimeMillis())
    val train = if (testPath.size > 3) data.cache() else splits(0).cache()
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



    val jedis = new Jedis("localhost")
    jedis.flushAll()
    val oHash = new ZFHash3(itqbitN, itqitN, itqratioN, upBound, splitN, isSparse)
    val objectData = train.mapPartitions(oHash.zfHashMap).persist(StorageLevel.MEMORY_AND_DISK)

//    val test = if (!isSparse) {
//      sc.textFile(testPath).map {
//        line =>
//          val parts = line.split(',')
//          new LabeledPoint(parts(parts.length - 1).toDouble, Vectors.dense(parts.take(parts.length - 1).map(_.toDouble)))
//      }
//    } else {
//      MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN)
//    }
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
    println("readT," + readT.sum + ",hashIterT," + hashIterT.sum + ",zipT," + zipT.sum + ",HashMapT," + HashMapT.sum + ",clusterN," + clusterN.size + ",/," + clusterN.sum + ",AVG," + clusterN.sum / clusterN.size + ",[," + clusterN.slice(0, 50).mkString(",") + ",],")
    println("itIncreSVDT," + itIncreSVDT.sum + ",itToBitT," + itToBitT.sum + ",itOthersT," + itOthersT.sum)
    jedis.close()


    val ratioL = if (test100) List(100) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
    //    val ratioL = if (test100) List(100) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    val vecs = new ArrayBuffer[Vector]()
    val mesb = new ArrayBuffer[Double]()
    val nntimesb = new ArrayBuffer[Double]()
    for (r <- ratioL) {
      val jedis = new Jedis("localhost")
      jedis.flushAll()
      val nnRatio = r / 100.0
      val train = objectData
      var trainN = 0.0
      val model: NonlinearModel = new NeuralNetworkModel(numFeature, hiddenNodesN)
      val modelTrain: ZFNNLSHPart3 = new ZFNNLSHPart3(model, train, nnRatio)
      val hissb = new StringBuilder()
      val w = w0.copy
      for (i <- 1 to nnItN) {
        val (g1, f1, itTrainN) = modelTrain.calculate(w, i)
        hissb.append("," + f1 / itTrainN)
        val itStepSize = stepSize / itTrainN / math.sqrt(i) //this is stepSize for each iteration
        w -= itStepSize * g1
        trainN += itTrainN
      }
      trainN /= nnItN
      vecs += Vectors.dense(w.toArray)
      val MSE = test.map { point =>
        val prediction = model.eval(w, new BDV[Double](point.features.toArray))
        (point.label, prediction)
      }.map { case (v, p) => 0.5 * math.pow((v - p), 2) }.mean()

      println()
      val zipN = jedis.get("zipN").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
      val setN = jedis.get("setN").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
      val nnMapT = jedis.get("nnMapT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)

      jedis.close()

      println(",nnRatio," + nnRatio + ",nnItN," + nnItN + ",initW," + w0(0) + ",step," + stepSize + ",hiddenN," + hiddenNodesN + ",trainN," + trainN / 10000.0 + ",testN," + test.count() / 10000.0 + ",numFeature," + numFeature)
      println("zipN," + zipN.sum / nnItN + ",setN," + setN.sum / nnItN + ",allUsedPointN," + trainN + ",nnMapT," + nnMapT.sum)
      System.out.println("w0= ," + w0.slice(0, math.min(w0.length, 10)).toArray.mkString(","))
      System.out.println("w = ," + w.slice(0, math.min(w.length, 10)).toArray.mkString(","))
      System.out.println(",MSE, " + MSE + ",[" + hissb)
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
