package AccurateML.lsh

import AccurateML.blas.{ZFBLAS, ZFUtils}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD
import edu.berkeley.cs.amplab.spark.indexedrdd.IndexedRDD._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangfan on 16/11/17.
  * this is old hash, layer-hash use ZFHashByIncreSVD
  */
class ZFHash(
               itqbitN: Int,
               itqitN: Int,
               itqratioN: Int,
               upBound: Int,
               spN: Double,
               sparse: Boolean,
               redisHost: String,
               sc: SparkContext = null
               //                           bcWeights:Broadcast[Vector]
             ) extends Serializable {

  val itqBitN = itqbitN
  val itqItN = itqitN
  val itqRatioN = itqratioN
  val upperBound = upBound
  val splitN = spN
  val featureVarianceIndex = new ArrayBuffer[Int]()
  val isSparse = sparse
  val mapT = if (sc != null) sc.longAccumulator else null

  /**
    * 根据方差直接换分属性
    * 递归划分
    */
  def directFeatureHashIterative(data: ArrayBuffer[LabeledPoint], indexs: Array[Int], hashIt: Int, bigN: String): mutable.HashMap[String, ArrayBuffer[Int]] = {

    var time = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)
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


  /**
    * svd产生itqBitN个新的属性,对每个新的属性划分成splitN份
    * 之后根据设定的upperBound值,迭代划分
    *
    */
  def hashIterative(data: ArrayBuffer[LabeledPoint], indexs: Array[Int], hashIt: Int, bigN: String): mutable.HashMap[String, ArrayBuffer[Int]] = {

    var time = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)
    val aMap = new mutable.HashMap[String, ArrayBuffer[Int]]()
    var numFeatures = data.last.features.size
    var numInc = indexs.size

    val zfsvd = new IncreSVD(data.toArray, indexs, itqBitN, itqItN, itqRatioN)
    zfsvd.calcFeaatures(isSparse)
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


  /**
    *
    * @param v      matrix[insN,featureN] contains new feature value from increSVD
    * @param u      be initialed as zero matrix, will be filled with 0 or 1
    * @param nfj    the new feature index, from 0 until nf. split one feature each splitU() function
    * @param indexs original index in a part
    */
  def splitU(v: BDM[Double], u: BDM[Int], nfj: Int, indexs: Array[Int]): Unit = {
    if (nfj >= v.cols)
      return
    val temp = new ArrayBuffer[Tuple2[Int, Double]]()
    for (i <- indexs) {
      temp += new Tuple2(i, v(i, nfj))
    }
    val tempIndex = temp.sortWith(_._2 < _._2).map(_._1)
    val halfSize = tempIndex.size / 2
    for (i <- halfSize until tempIndex.size) {
      u(tempIndex(i), nfj) = 1
    }
    val index0 = tempIndex.slice(0, halfSize).toArray
    val index1 = tempIndex.slice(halfSize, tempIndex.size).toArray
    splitU(v, u, nfj + 1, index0)
    splitU(v, u, nfj + 1, index1)
  }

  /**
    * 2017-7-12
    * 一次生成多个属性nf=itqBitN,逐个进一步数量上二分,IncreSVD的第三个参数设置nf
    */
  def hash3Once(data: Array[LabeledPoint], indexs: Array[Int], hashIt: Int): mutable.HashMap[String, ArrayBuffer[Int]] = {

    var time = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)
    val aMap = new mutable.HashMap[String, ArrayBuffer[Int]]()
    var numFeatures = data.last.features.size
    var numInc = indexs.size

    val tempItqBitN = math.log(indexs.size / upperBound) / math.log(2)
    val zfsvd = new IncreSVD(data.toArray, indexs, math.round(tempItqBitN).toInt, itqItN, itqRatioN)
    //    val zfsvd = new IncreSVD(data, indexs, itqBitN, itqItN, itqRatioN)
    zfsvd.calcFeaatures(isSparse)
    val v: BDM[Double] = zfsvd.userFeas.t
    jedis.append("itIncreSVDT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val u = BDM.zeros[Int](v.rows, v.cols)

    splitU(v, u, 0, Array.range(0, v.rows))
    jedis.append("itToBitT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    for (i <- 0 until u.rows) {
      val key = u(i, ::).inner.toArray.mkString("")
      //      val key = u(i, ::).inner.toArray.mkString(",")
      val mapKey = key
      val aset = aMap.getOrElse(mapKey, new ArrayBuffer[Int]())
      aset += indexs(i)
      aMap.update(mapKey, aset)
    }
    jedis.append("itOthersT", "," + (System.currentTimeMillis() - time))

    jedis.close()
    aMap
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
    val jedis = new Jedis(redisHost)

    var time = System.currentTimeMillis()
    var partData = pit.toArray
    jedis.append("readT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val numFeatures = partData.last.features.size
    val indexs = Array.range(0, partData.size)
    //    val aMap = directFeatureHashIterative(partData, indexs, 0, "")
    val aMap = hash3Once(partData, indexs, 0)
    //    val aMap = hashIterative(partData, indexs, 0, "")
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
    mapT.add(System.currentTimeMillis() - HashMapT)
    jedis.append("HashMapT", "," + (System.currentTimeMillis() - HashMapT))
    jedis.close()
    ansAll.iterator
  }


  def zfHashMap(pit: Iterator[LabeledPoint]): Iterator[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])] = {

    val HashMapT = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)

    var time = System.currentTimeMillis()
    var partData = pit.toArray
    jedis.append("readT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val numFeatures = partData.last.features.size
    val indexs = Array.range(0, partData.size)
    val aMap = hash3Once(partData, indexs, 0)
    //    val aMap = hashIterative(partData, indexs, 0, "")
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
    mapT.add((System.currentTimeMillis() - HashMapT))
    jedis.close()
    ansAll.iterator
  }

  def zfHashMapIndexedRDD(pit: Iterator[(Long, (LabeledPoint, Double))]): Iterator[(String, Array[Long])] = {

    val HashMapT = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)

    var time = System.currentTimeMillis()
    val longIndex = new ArrayBuffer[Long]()
    val partData = new ArrayBuffer[LabeledPoint]()
    while (pit.hasNext) {
      val temp = pit.next()
      longIndex += temp._1
      partData += temp._2._1
    }
    jedis.append("readT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val numFeatures = partData.last.features.size
    val indexs = Array.range(0, partData.size)
    val aMap = hash3Once(partData.toArray, indexs, 0)
    //    val aMap = hashIterative(partData, indexs, 0, "")
    jedis.append("hashIterT", "," + (System.currentTimeMillis() - time))


    mapT.add((System.currentTimeMillis() - HashMapT))
    jedis.close()
    aMap.toArray.map(t2 => (t2._1, t2._2.toArray.map(i => longIndex(i)))).toIterator
  }

  def zfHashMapIndexedRDD1(pit: Iterator[(Long, (LabeledPoint, Double))]): Iterator[Array[Array[Long]]] = {

    val HashMapT = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)

    var time = System.currentTimeMillis()
    val partObjectData = pit.toArray
    val longIndex = partObjectData.map(_._1)
    val partData = partObjectData.map(_._2._1)
    jedis.append("readT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val aMap = hash3Once(partData.toArray, Array.range(0, partData.size), 0)
    val index1: Iterable[Array[Int]] = aMap.values.map(_.toArray) //在一个part里,将数据分成多个Array
    val index2: Iterable[Array[Array[Int]]] = index1.map { is => hash3Once(partData.toArray, is, 0).values.map(_.toArray).toArray }

    jedis.append("hashIterT", "," + (System.currentTimeMillis() - time))
    mapT.add((System.currentTimeMillis() - HashMapT))
    jedis.close()
    index2.map(aa => aa.map(is => is.map(i => longIndex(i)))).iterator
  }


}


object ZFHash {
  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val logFile = "README.md" // Should be some file on your system
    val conf = new SparkConf().setAppName("github.KernelSVM Test")
    //    val conf = new SparkConf().setAppName("github.KernelSVM Test").setMaster("local")
    val sc = new SparkContext(conf)


    ///Users/zhangfan/Documents/data/a9a - 0 0 10,50,100 true 123 1
    val dataPath = "/Users/zhangfan/Downloads/c"
    val isSparse = true
    val numFeature = 3
    val minPartN = 1

    val itqbitN = 2
    val itqitN = 10
    val itqratioN = 1
    val upBound = 3
    val redisHost = "localhost"


    val splitChar = ","

    val data: RDD[LabeledPoint] = if (!isSparse) {
      sc.textFile(dataPath, minPartN).map(line => {
        val vs = line.split(splitChar).map(_.toDouble)
        val features = vs.slice(0, vs.size - 1)
        LabeledPoint(vs.last, Vectors.dense(features).toSparse.asInstanceOf[SparseVector]) // must be sparse
      })
    } else {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
    }
    val cancel1 = data.collect()

    val zipTime = System.currentTimeMillis()
    val indexedData: IndexedRDD[Long, (LabeledPoint, Double)] = IndexedRDD(data.zipWithUniqueId().map { case (k, v) => (v, (k, 0D)) })
    val c0 = indexedData.collect()

    println()

  }
}