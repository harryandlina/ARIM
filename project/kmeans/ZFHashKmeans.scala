package AccurateML.kmeans

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis
import AccurateML.blas.{ZFBLAS, ZFUtils}
import AccurateML.nonLinearRegression.ZFHash3

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by zhangfan on 17/1/12.
  */


class ZFHashKmeans(k: Int, itN: Int) extends Serializable {

  val costHis = new ArrayBuffer[Double]()


  def runGradientAlgorithm(data: RDD[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])], origCenters: Array[Vector], ratio: Double): Array[Vector] = {
    if (data.getStorageLevel == StorageLevel.NONE) {
      System.err.println("------------------The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.------------------------")
    }
    val sc = data.sparkContext
    val mapdata = data.mapPartitions(objectPoints => {
      val jedis = new Jedis("localhost")
      val nnMapT = System.currentTimeMillis()
      val ans = new ArrayBuffer[Tuple2[Tuple2[LabeledPoint, Array[Double]], ArrayBuffer[Tuple2[LabeledPoint, Array[Double]]]]]()

      while (objectPoints.hasNext) {
        val objectP = objectPoints.next()
        val zip: LabeledPoint = objectP._1(0).last
        val points = objectP._1(1)
        val mapPoints = new ArrayBuffer[Tuple2[LabeledPoint, Array[Double]]]()
        points.foreach(point => {
          mapPoints += Tuple2(point, Array(0.0, 0.0))
        })
        ans += Tuple2(Tuple2(zip, Array(0.0, 0.0)), mapPoints) // (lastLabel,lastCost)
      }
      jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
      ans.iterator
    }).cache()

    data.unpersist()

    val centers = origCenters.clone()

    for (it <- 0 until itN) {
      if (it == 0) {
        val bcCenters = sc.broadcast(centers)
        val costAccum = sc.doubleAccumulator

        val totalContibs = mapdata.mapPartitions { mapPoints =>
          val jedis = new Jedis("localhost")
          val nnMapT = System.currentTimeMillis()
          val thisCenters = bcCenters.value
          val dims = thisCenters(0).size
          val sums = Array.fill(k)(Vectors.zeros(dims))
          val counts = Array.fill(k)(0L)
          mapPoints.foreach { mapP =>
            val zipTuple = mapP._1
            val (centerIndex, cost) = ZFHashKmeans.zfFindClosest(zipTuple._1.features, thisCenters)
            zipTuple._2(0) = centerIndex
            zipTuple._2(1) = cost

            val pointsTuple = mapP._2
            pointsTuple.foreach(pt => {
              val (centerIndex, cost) = ZFHashKmeans.zfFindClosest(pt._1.features, thisCenters)
              counts(centerIndex) += 1
              costAccum.add(cost)
              ZFBLAS.axpy(1.0, pt._1.features, sums(centerIndex))
              pt._2(0) = centerIndex
              pt._2(1) = cost
            })
          }
          val contribs = for (i <- 0 until k) yield {
            (i, (sums(i), counts(i)))
          }

          val partN = counts.sum
          jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
          jedis.append("partN", "," + partN)
          jedis.close()
          contribs.iterator
        }.reduceByKey((a, b) => {
          ZFBLAS.axpy(1.0, a._1, b._1)
          (b._1, a._2 + b._2)
        }).collectAsMap()
        bcCenters.unpersist(blocking = false)
        for (i <- 0 until k) {
          val (sum, count) = totalContibs(i)
          if (count != 0) {
            ZFBLAS.scal(1.0 / count, sum)
            centers(i) = sum
          }
        }
        costHis += costAccum.value
      } else {
        // it>0
        val bcCenters = sc.broadcast(centers)
        val costAccum = sc.doubleAccumulator

        val totalContibs = mapdata.mapPartitions { mapPoints =>
          val (it1, it2) = mapPoints.duplicate
          val jedis = new Jedis("localhost")
          val nnMapT = System.currentTimeMillis()
          val thisCenters = bcCenters.value
          val dims = thisCenters(0).size
          val sums = Array.fill(k)(Vectors.zeros(dims))
          val counts = Array.fill(k)(0L)

          var cenCostAB = new ArrayBuffer[Tuple4[Int, Double, Double, Int]]() //centerIndex,cost,gap,zipIndex
          it1.zipWithIndex.foreach { mapP =>
            val zipTuple = mapP._1._1
            val zipIndex = mapP._2
            val (centerIndex, cost) = ZFHashKmeans.zfFindClosest(zipTuple._1.features, thisCenters)
            val gap = math.abs(cost - zipTuple._2(1))
            zipTuple._2(0) = centerIndex
            zipTuple._2(1) = cost
            cenCostAB += Tuple4(centerIndex, cost, gap, zipIndex)
          }

          val ratioN = (cenCostAB.size * ratio).toInt
          val zipSortIndex = cenCostAB.sortWith(_._3 > _._3).slice(0, ratioN).map(_._4) //.slice(0, ratioN)
        var tempi = 0
          while (it2.hasNext) {
            val (_, pointsTuple) = it2.next()
            if (zipSortIndex.contains(tempi)) {
              pointsTuple.foreach(pt => {
                val (centerIndex, cost) = ZFHashKmeans.zfFindClosest(pt._1.features, thisCenters)
                pt._2(0) = centerIndex
                pt._2(1) = cost
              })
            }
            pointsTuple.foreach(pt => {
              val centerIndex = pt._2(0).toInt
              val cost = pt._2(1)
              counts(centerIndex) += 1
              costAccum.add(cost)
              ZFBLAS.axpy(1.0, pt._1.features, sums(centerIndex))
            })
            tempi += 1
          }

          val contribs = for (i <- 0 until k) yield {
            (i, (sums(i), counts(i)))
          }

          val partN = counts.sum
          jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
          jedis.append("partN", "," + partN)
          jedis.close()
          contribs.iterator
        }.reduceByKey((a, b) => {
          ZFBLAS.axpy(1.0, a._1, b._1)
          (b._1, a._2 + b._2)
        }).collectAsMap()
        bcCenters.unpersist(blocking = false)
        for (i <- 0 until k) {
          val (sum, count) = totalContibs(i)
          if (count != 0) {
            ZFBLAS.scal(1.0 / count, sum)
            centers(i) = sum
          }
        }
        costHis += costAccum.value
      }

    } //end-it
    centers
  }


  def zfComputeCost(data: RDD[Vector], centers: Array[Vector]): Double = {
    val ans = data.map(point => ZFHashKmeans.zfFindClosest(point, centers)._2).sum()
    ans
  }

}


object ZFHashKmeans {
  def zfFindClosest(point: Vector, centers: Array[Vector]): Tuple2[Int, Double] = {
    var minIndex: Int = -1
    var minValue: Double = Double.MaxValue
    for (i <- 0 until centers.size) {

      if (point.size != centers(i).size) {
        println(point.size + ",\t" + centers(i).size)
        println("diff")
      }

      var diff: Vector = centers(i).copy
      diff match {
        case dy: DenseVector =>
          ZFBLAS.axpy(-1, point, dy)
        case sy: SparseVector => {
          val temp: DenseVector = sy.toDense
          ZFBLAS.axpy(-1, point, temp)
          diff = temp.toSparse
        }
      }
      val cost = ZFBLAS.dot(diff, diff)
      if (cost < minValue) {
        minValue = cost
        minIndex = i
      }
    }
    Tuple2(minIndex, minValue)
  }

  def testKMeans(sc: SparkContext): Unit = {
    val data = sc.textFile("/Users/zhangfan/Documents/data/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    // Save and load model
    //    clusters.save(sc, "target/org/apache/spark/KMeansExample/KMeansModel")
    //    val sameModel = KMeansModel.load(sc, "target/org/apache/spark/KMeansExample/KMeansModel")
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)

    val k = args(0).toInt //2
    val itN = args(1).toInt //5
    val numFeatures = args(2).toInt //102660
    val centerPath = args(3)
    val dataPath = args(4) //"/Users/zhangfan/Documents/data/kmeans_data.txt"
    val test100 = args(5).toBoolean
    val isSparse = args(6).toBoolean
    val minPartN = args(7).toInt

    val itqbitN = args(8).toInt
    val itqitN = args(9).toInt
    val itqratioN = args(10).toInt //from 1 not 0
    val upBound = args(11).toInt
    val splitN = args(12).toDouble


    val data: RDD[LabeledPoint] = if (isSparse) {
      MLUtils.loadLibSVMFile(sc, dataPath, numFeatures, minPartN)
    } else {
      sc.textFile(dataPath, minPartN).map(s => new LabeledPoint(0.0, Vectors.dense(s.split(",").map(_.toDouble))))
    }
    val jedis = new Jedis("localhost")
    jedis.flushAll()
    val oHash = new ZFHash3(itqbitN, itqitN, itqratioN, upBound, splitN, isSparse)
    val objectData = data.mapPartitions(oHash.zfHashMap).persist(StorageLevel.MEMORY_AND_DISK)

    val on = objectData.count()
    println()
    println("dataPart," + data.getNumPartitions + ",objectDataPart," + objectData.getNumPartitions)
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


    val rDivN = 100
    val ratioL = if (test100) List(rDivN) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    //    val ratioL = if (test100) List(rDivN) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60)
    val mesb = new ArrayBuffer[Double]()
    val nntimesb = new ArrayBuffer[Double]()
    for (r <- ratioL) {
      val ratio = r / rDivN.toDouble
      val jedis = new Jedis("localhost")
      jedis.flushAll()
      val train = objectData
      val origCenters = new Array[Vector](k)
      val iter = Source.fromFile(centerPath).getLines()
      var tempk = 0
      while (iter.hasNext && tempk < k) {
        val line = iter.next()
        if (!isSparse) {
          origCenters(tempk) = Vectors.dense(line.split(",").map(_.toDouble))
        } else {
          val vs = line.split("\\s")
          val features = vs.slice(1, vs.size).map(s => s.split(":"))
          val indexs = features.map(arr => arr(0).toInt)
          val values = features.map(arr => arr(1).toDouble)
          origCenters(tempk) = Vectors.sparse(numFeatures, indexs, values)
        }
        tempk += 1
      }
      val zfkmeans = new ZFHashKmeans(k, itN)
      val centers = zfkmeans.runGradientAlgorithm(train, origCenters, ratio)

      val partN = jedis.get("partN").split(",")
      val nnMapT = jedis.get("nnMapT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
      val WSSSE = zfkmeans.zfComputeCost(data.map(point => point.features), centers)
      mesb += WSSSE
      nntimesb += nnMapT.sum
      println()
      println("dataPart," + data.getNumPartitions + ",objectDataPart," + objectData.getNumPartitions + ",trainPart," + train.getNumPartitions + ",nnMapT," + nnMapT.sum + ",origCenter," + origCenters.map(vec => vec.apply(0)).slice(0, 5).mkString(","))
      //      println("partN," + partN.slice(0, math.min(partN.length, 20)).mkString(","))
      println(",ratio," + ratio + ",k," + k + ",itN," + itN + ",trainN," + train.count() / 10000.0 + ",testN," + data.count() / 10000.0 + ",numFeatures," + data.first().features.size)
      System.out.println(",WSSSE, " + WSSSE + ",[" + zfkmeans.costHis.mkString(","))
      jedis.close()
      train.unpersist()
    }
    println()
    println(this.getClass.getName + ",data," + dataPath)
    println("ratio,MSE,nnMapT")
    for (i <- ratioL.indices) {
      println(ratioL(i) / rDivN.toDouble + "," + mesb(i) + "," + nntimesb(i))
    }

  }
}
