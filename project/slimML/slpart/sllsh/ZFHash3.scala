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
import AccurateML.lsh.IncreSVD2

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
    //    val jedis = new Jedis("localhost")
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
    //    jedis.append("itIncreSVDT", "," + (System.currentTimeMillis() - time))


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
    //    jedis.append("itToBitT", "," + (System.currentTimeMillis() - time))

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
    //    jedis.append("itOthersT", "," + (System.currentTimeMillis() - time))

    for (temp <- aMap.toList.sortWith(_._1 < _._1)) {
      if (temp._2.size <= upperBound || hashIt >= 50) {
        ans += temp
      } else {
        ans ++= directFeatureHashIterative(data, temp._2.toArray, hashIt + 1, bigN + bN)
        bN += 1
      }
    }
    //    jedis.close()
    ans
  }


  def hashIterative(data: ArrayBuffer[LabeledPoint], indexs: Array[Int], hashIt: Int, bigN: String): mutable.HashMap[String, ArrayBuffer[Int]] = {

    var time = System.currentTimeMillis()
    //    val jedis = new Jedis("localhost")
    val aMap = new mutable.HashMap[String, ArrayBuffer[Int]]()
    var numFeatures = data.last.features.size
    var numInc = indexs.size

    val zfsvd = new IncreSVD2(data, indexs, itqBitN, itqItN, itqRatioN)
    zfsvd.calcFeaatures(hashIt, isSparse)
    val v: BDM[Double] = zfsvd.userFeas.t
    //    jedis.append("itIncreSVDT", "," + (System.currentTimeMillis() - time))

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
    //    jedis.append("itToBitT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val prefix: String = bigN + "@"
    for (i <- 0 until u.rows) {
      val key = u(i, ::).inner.toArray.mkString(",")
      val mapKey = key + prefix



      val aset = aMap.getOrElse(mapKey, new ArrayBuffer[Int]())
      aset += indexs(i)
      aMap.update(mapKey, aset)

      //      println(s"mapKey: ${mapKey} aset.size: ${aset.size}")
    }
    var bN: Int = 0
    val ans = new mutable.HashMap[String, ArrayBuffer[Int]]()
    //    jedis.append("itOthersT", "," + (System.currentTimeMillis() - time))

    for (temp <- aMap.toList.sortWith(_._1 < _._1)) {
      if (temp._2.size <= upperBound || hashIt >= 50) {
        ans += temp
      } else {
        ans ++= hashIterative(data, temp._2.toArray, hashIt + 1, bigN + bN)
        bN += 1
      }
    }
    //    jedis.close()
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
    //    val jedis = new Jedis("localhost")

    var time = System.currentTimeMillis()
    var partData = new ArrayBuffer[LabeledPoint]()
    while (pit.hasNext) {
      partData += pit.next()
    }
    //    jedis.append("readT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val numFeatures = partData.last.features.size
    val indexs = Array.range(0, partData.size)
    //    val aMap = directFeatureHashIterative(partData, indexs, 0, "")
    val aMap = hashIterative(partData, indexs, 0, "")
    //    jedis.append("hashIterT", "," + (System.currentTimeMillis() - time))

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
      //      jedis.append("clusterN", "," + points.size)
      ansAll += Tuple3(zips, Array[ArrayBuffer[(Int, Int)]](), new ArrayBuffer[mutable.HashSet[Int]])
    }
    //    jedis.append("zipT", "," + (System.currentTimeMillis() - time))
    //    jedis.append("HashMapT", "," + (System.currentTimeMillis() - HashMapT))
    //    jedis.close()
    ansAll.iterator
  }


  def zfHashMap(pit: Iterator[LabeledPoint]): Iterator[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])] = {
    val HashMapT = System.currentTimeMillis()
    //    val jedis = new Jedis("localhost")

    var time = System.currentTimeMillis()
    var partData = new ArrayBuffer[LabeledPoint]()
    while (pit.hasNext) {
      partData += pit.next()
    }
    //    jedis.append("readT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val numFeatures = partData.last.features.size
    val indexs = Array.range(0, partData.size)
    //    val aMap = directFeatureHashIterative(partData, indexs, 0, "")
    val aMap = hashIterative(partData, indexs, 0, "")
    //    jedis.append("hashIterT", "," + (System.currentTimeMillis() - time))

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
      //      jedis.append("clusterN", "," + points.size)
      ansAll += Tuple3(zips, Array[ArrayBuffer[(Int, Int)]](), new ArrayBuffer[mutable.HashSet[Int]])
    }
    //    jedis.append("zipT", "," + (System.currentTimeMillis() - time))
    //    jedis.append("HashMapT", "," + (System.currentTimeMillis() - HashMapT))
    //    jedis.close()
    ansAll.iterator
  }


}