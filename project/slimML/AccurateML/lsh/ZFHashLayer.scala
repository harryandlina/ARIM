package AccurateML.lsh

import AccurateML.blas.{ZFBLAS, ZFBLASml}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{LabeledPoint => mlLabeledPoint}
import org.apache.spark.ml.linalg.{Vector => mlVector, Vectors => mlVectors}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangfan on 18/1/11. ZFHashByIncreSVD
  */

class ZFHashLayer(
                   itqitN: Int,
                   itqratioN: Int,
                   upbound: Int,
                   sparse: Boolean,
                   layer: Int = 2, //压缩点分几层
                   redisHost: String,
                   sc: SparkContext = null,
                   labelCount: Int = 0
                   //                           bcWeights:Broadcast[Vector]
                 ) extends Serializable {


  val itqItN = itqitN
  val itqRatioN = itqratioN
  val upBound = upbound
  val isSparse = sparse
  val mapT = if (sc != null) sc.longAccumulator else null

  def zfHash(pit: Iterable[LabeledPoint]): Iterator[(LabeledPoint, Array[LabeledPoint])] = {

    val HashMapT = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)

    val partData: Array[LabeledPoint] = pit.toArray
    val numFeature = partData.last.features.size
    val firstIndexAB: Array[Array[Int]] = hashOnce(partData, Array.range(0, partData.size), upBound)

    val ans = firstIndexAB.filter(_.size > 0).map(is => {
      val zipFeature1 = Vectors.zeros(numFeature)
      var label1 = 0.0
      val points = is.map(i => partData(i))
      points.foreach(p => {
        ZFBLAS.axpy(1.0, p.features, zipFeature1)
        label1 += p.label
      })
      ZFBLAS.scal(1.0 / is.size, zipFeature1)
      label1 = label1 / is.size
      (new LabeledPoint(label1, zipFeature1), points)
    })

    mapT.add((System.currentTimeMillis() - HashMapT))
    jedis.close()
    ans.iterator
  }

  def zfHashMapMLPC(pit: Iterable[mlLabeledPoint]): Iterator[(Vector, Array[mlLabeledPoint])] = {

    val HashMapT = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)

    val partData: Array[mlLabeledPoint] = pit.toArray
    val numFeature = partData.last.features.size
    val firstIndexAB: Array[Array[Int]] = hashOnceMLPC(partData, Array.range(0, partData.size), upBound)

    val ans = firstIndexAB.map(is => {
      val zipFeature1 = mlVectors.zeros(numFeature)
      var label1 = 0.0
      val points = is.map(i => partData(i))
      points.foreach(p => {
        ZFBLASml.axpy(1.0, p.features, zipFeature1)
        label1 += p.label
      })
      ZFBLASml.scal(1.0 / is.size, zipFeature1)
      label1 = label1 / is.size
      val zip1 = new mlLabeledPoint(label1, zipFeature1)

      //      val vecZip:(mlVector, mlVector) = ZFLabelConverter.encodeLabeledPoint(zip1, 3)
      //      val stackZip = Vectors.fromBreeze(BDV.vertcat(
      //        vecZip._1.asBreeze.toDenseVector,
      //        vecZip._2.asBreeze.toDenseVector))
      val output = Array.fill(labelCount)(0.0)
      output(label1.toInt) = 1.0
      val stackZip = Vectors.dense(zipFeature1.toArray ++ output)
      (stackZip, points.map(lp => new mlLabeledPoint(lp.label, lp.features)))
    })

    mapT.add((System.currentTimeMillis() - HashMapT))
    jedis.close()
    ans.iterator
  }


  def zfHashMap(pit: Iterator[LabeledPoint]): Iterator[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[Array[Int]]])] = {

    val HashMapT = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)

    var time = System.currentTimeMillis()
    val partData: Array[LabeledPoint] = pit.toArray
    jedis.append("readT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val numFeatures = partData.last.features.size
    val firstIndexAB: Array[Array[Int]] = hashOnce(partData, Array.range(0, partData.size), upBound)
    val firstIncs: Array[ArrayBuffer[LabeledPoint]] = firstIndexAB.map(is => {
      val ab = new ArrayBuffer[LabeledPoint]()
      is.foreach(i => ab += partData(i))
      ab
    })
    val ans = new ArrayBuffer[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[Array[Int]]])]()
    for (ins <- firstIncs) {
      jedis.append("clusterN", "," + ins.size)
      var indexAB = new ArrayBuffer[Array[Int]]
      indexAB += Array.range(0, ins.size)
      val ansZip = new Array[ArrayBuffer[LabeledPoint]](layer + 1) //last layer is original points( not zip )
      val ansIndex = new Array[ArrayBuffer[Array[Int]]](layer)
      ansZip(0) = ArrayBuffer.fill(1)(makeZip(ins, Array.range(0, ins.size)))
      ansIndex(0) = ArrayBuffer.fill(1)(Array.range(0, ins.size)) /// 0_zipIndex is all indexs
      ansZip(layer) = ins

      for (l <- 1 until layer) {
        indexAB = indexAB.flatMap(indexs => {
          hashOnce(ins.toArray, indexs, upBound / math.pow(4, l))
        })
        ansIndex(l) = indexAB // avoid reference change the value
        ansZip(l) = indexAB.map(makeZip(ins, _))
      }
      ansZip(layer) = ins
      ans += Tuple2(ansZip, ansIndex)
    }

    mapT.add((System.currentTimeMillis() - HashMapT))
    jedis.close()
    ans.iterator
  }

  def zfHashMapIndexedRDD2(pit: Iterator[(Long, (LabeledPoint, Double))]): Iterator[Array[(LabeledPoint, Array[Long])]] = {

    val HashMapT = System.currentTimeMillis()
    val jedis = new Jedis(redisHost)
    var time = System.currentTimeMillis()
    val partObjectData = pit.toArray
    val longIndex = partObjectData.map(_._1)
    val partData = partObjectData.map(_._2._1)
    jedis.append("readT", "," + (System.currentTimeMillis() - time))

    time = System.currentTimeMillis()
    val index1: Array[Array[Int]] = hashOnce(partData.toArray, Array.range(0, partData.size), upBound) //在一个part里,将数据分成多个Array
    val index2: Array[Array[Array[Int]]] = index1.map { is => hashOnce(partData.toArray, is, upbound / 2.0) }

    jedis.append("hashIterT", "," + (System.currentTimeMillis() - time))
    mapT.add((System.currentTimeMillis() - HashMapT))
    jedis.close()
    val numFeature = partData.last.features.size
    val ans: Iterable[Array[(LabeledPoint, Array[Long])]] = index2.map(aa => {
      val zipFeature0 = Vectors.zeros(numFeature)
      var count0 = 0
      var label0 = 0.0
      val aaMap = aa.map(is => {
        val zipFeature1 = Vectors.zeros(numFeature)
        var label1 = 0.0
        is.foreach(i => {
          ZFBLAS.axpy(1.0, partData(i).features, zipFeature1)
          label1 += partData(i).label
        })
        ZFBLAS.axpy(1.0, zipFeature1, zipFeature0)

        count0 += is.size
        label0 += label1
        ZFBLAS.scal(1.0 / is.size, zipFeature1)

        val zip1 = if (isSparse) new LabeledPoint(label1 / is.size, zipFeature1.toSparse) else new LabeledPoint(label1 / is.size, zipFeature1)
        (zip1, is.map(i => longIndex(i)))
      })
      ZFBLAS.scal(1.0 / count0, zipFeature0)
      val zip0 = if (isSparse) new LabeledPoint(label0 / count0, zipFeature0.toSparse) else new LabeledPoint(label0 / count0, zipFeature0)
      Array((zip0, Array[Long]())) ++ aaMap
    })
    ans.iterator
  }

  /**
    *
    * @param points
    * @param index
    * @return zip with label 0
    */
  def makeZip(points: ArrayBuffer[LabeledPoint], index: Array[Int]): LabeledPoint = {
    val numFeatures = points.last.features.size
    val zipfea = Vectors.zeros(numFeatures)
    for (i <- index) {
      val point = points(i)
      ZFBLAS.axpy(1.0, point.features, zipfea)
    }
    val divn = 1.0 / index.size
    ZFBLAS.scal(divn, zipfea)
    val zip = if (!isSparse) new LabeledPoint(0, zipfea) else new LabeledPoint(0, zipfea.toSparse)
    zip
  }


  def hashOnce(data: Array[LabeledPoint], indexs: Array[Int], tempUpperBound: Double): Array[Array[Int]] = {
    if (indexs.size <= 1) {
      return Array(indexs)
    }
    var time = System.currentTimeMillis()
    val aMap = new mutable.HashMap[String, ArrayBuffer[Int]]()
    var numFeatures = data.last.features.size
    var numInc = indexs.size

    val tempItqBitN = math.log(indexs.size / tempUpperBound) / math.log(2)
    val zfsvd = new IncreSVD(data, indexs, math.round(tempItqBitN).toInt, itqItN, itqRatioN)
    zfsvd.calcFeaatures(isSparse)
    val v: BDM[Double] = zfsvd.userFeas.t
    time = System.currentTimeMillis()
    val u = BDM.zeros[Int](v.rows, v.cols)

    splitU(v, u, 0, Array.range(0, v.rows))

    time = System.currentTimeMillis()
    for (i <- 0 until u.rows) {
      val key = u(i, ::).inner.toArray.mkString("")
      val mapKey = key
      val aset = aMap.getOrElse(mapKey, new ArrayBuffer[Int]())
      aset += indexs(i)
      aMap.update(mapKey, aset)
    }
    aMap.map(_._2.toArray).toArray
  }


  def hashOnceMLPC(data: Array[mlLabeledPoint], indexs: Array[Int], tempUpperBound: Double): Array[Array[Int]] = {
    if (indexs.size <= 1) {
      return Array(indexs)
    }
    var time = System.currentTimeMillis()
    val aMap = new mutable.HashMap[String, ArrayBuffer[Int]]()
    var numFeatures = data.last.features.size
    var numInc = indexs.size

    val tempItqBitN = math.log(indexs.size / tempUpperBound) / math.log(2)
    val zfsvd = new IncreSVDMLPC(data, indexs, math.round(tempItqBitN).toInt, itqItN, itqRatioN)
    zfsvd.calcFeaatures(isSparse)
    val v: BDM[Double] = zfsvd.userFeas.t
    time = System.currentTimeMillis()
    val u = BDM.zeros[Int](v.rows, v.cols)

    splitU(v, u, 0, Array.range(0, v.rows))

    time = System.currentTimeMillis()
    for (i <- 0 until u.rows) {
      val key = u(i, ::).inner.toArray.mkString("")
      val mapKey = key
      val aset = aMap.getOrElse(mapKey, new ArrayBuffer[Int]())
      aset += indexs(i)
      aMap.update(mapKey, aset)
    }
    aMap.map(_._2.toArray).toArray
  }

  /**
    *
    * @param v      input matrix
    * @param u      01 matrix according to v
    * @param nfj    split v value index
    * @param vIndex which instances to be split in this round
    */

  def splitU(v: BDM[Double], u: BDM[Int], nfj: Int, vIndex: Array[Int]): Unit = {
    if (nfj >= v.cols)
      return
    //    val cancel = vIndex.sortWith(_ < _)
    //    println("splitU() v (" + v.rows + "," + v.cols + " ) \t nfj" + nfj + "\t indexs: " + vIndex.size + " [" + cancel(0) + "," + cancel.last)

    val temp = new ArrayBuffer[Tuple2[Int, Double]]()
    for (i <- vIndex) {
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


}

object ZFHashLayer {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)

    //2 3 3 /Users/zhangfan/Documents/data/kmeans_data.txt /Users/zhangfan/Documents/data/kmeans_data.txt 100 false 1 eucli 10 1 4 - localhost
    val k = args(0).toInt //2
    val itN = args(1).toInt //5
    val numFeatures = args(2).toInt //102660
    val centerPath = args(3)
    val dataPath = args(4) //"/Users/zhangfan/Documents/data/kmeans_data.txt"
    val test100: Array[Double] = args(5).split(",").map(_.toDouble)
    val isSparse = args(6).toBoolean
    val minPartN = args(7).toInt
    val disFunc = args(8)

    val itqitN = args(9).toInt
    val itqratioN = args(10).toInt //from 1 not 0
    val upBound = args(11).toInt
    val testPath = args(12)
    val hashLayer = args(13).toInt //"172.18.11.97"


    val data: RDD[LabeledPoint] = if (isSparse) {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeatures, minPartN)
    } else {
      sc.textFile(dataPath, minPartN).map(s => {
        val vs = s.split("\\s+|,")
        new LabeledPoint(0.0, Vectors.dense(vs.slice(0, vs.size - 1).map(_.toDouble)))
      })
    }


    //    val oHash = new ZFHashByIncreSVD(itqitN, itqratioN, upBound, isSparse, hashLayer, redisHost,sc)
    //    val objectData: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[Array[Int]]])] = data
    //      .mapPartitions(oHash.zfHashMap) //incrementalSVD
    //      .persist(StorageLevel.MEMORY_AND_DISK)
    //    val on = objectData.count()


  }
}
