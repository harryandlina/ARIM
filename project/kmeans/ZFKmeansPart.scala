package AccurateML.kmeans

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis
import AccurateML.blas.{ZFBLAS, ZFUtils}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by zhangfan on 17/1/9.
  */


class ZFKmeansPart(k: Int, itN: Int) extends Serializable {

  val costHis = new ArrayBuffer[Double]()


  def runGradientAlgorithm(data: RDD[Vector], origCenters: Array[Vector], ratio: Double): Array[Vector] = {
    if (data.getStorageLevel == StorageLevel.NONE) {
      System.err.println("------------------The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.------------------------")
    }
    val sc = data.sparkContext
    val mapdata = data.mapPartitions(points => {
      val jedis = new Jedis("localhost")
      val nnMapT = System.currentTimeMillis()

      val ans = new ArrayBuffer[Tuple2[Vector, Array[Double]]]()
      while (points.hasNext) {
        ans += Tuple2(points.next(), Array(0.0, 0.0)) // (lastLabel,lastCost)
      }
      jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
      ans.iterator
    }).cache()

    val centers = origCenters.clone()

    for (it <- 0 until itN) {
      if (it == 0) {
        val bcCenters = sc.broadcast(centers)
        val costAccum = sc.doubleAccumulator

        val totalContibs = mapdata.mapPartitions { points =>
          val jedis = new Jedis("localhost")
          val nnMapT = System.currentTimeMillis()
          val thisCenters = bcCenters.value
          val dims = thisCenters(0).size
          val sums = Array.fill(k)(Vectors.zeros(dims))
          val counts = Array.fill(k)(0L)
          points.foreach { point =>
            val (centerIndex, cost) = ZFKmeansPart.zfFindClosest(point._1, thisCenters)
            counts(centerIndex) += 1
            costAccum.add(cost)
            ZFBLAS.axpy(1.0, point._1, sums(centerIndex))
            point._2(0) = centerIndex
            point._2(1) = cost
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

        val totalContibs = mapdata.mapPartitions { points =>
          val jedis = new Jedis("localhost")
          val nnMapT = System.currentTimeMillis()
          val thisCenters = bcCenters.value
          val dims = thisCenters(0).size
          val sums = Array.fill(k)(Vectors.zeros(dims))
          val counts = Array.fill(k)(0L)

          var cenCostAB = new ArrayBuffer[Tuple4[Int, Double, Double, Tuple2[Vector, Array[Double]]]]()

          points.foreach { point =>
            val (centerIndex, cost) = ZFKmeansPart.zfFindClosest(point._1, thisCenters)
            val gap = math.abs(cost - point._2(1))
            cenCostAB += Tuple4(centerIndex, cost, gap, point)
          }

          val ratioN = (cenCostAB.size * ratio).toInt
          cenCostAB = cenCostAB.sortWith(_._3 > _._3) //.slice(0, ratioN)
          for (i <- 0 until cenCostAB.size) {
            if (i < ratioN) {
              cenCostAB(i) match {
                case (centerIndex, cost, gap, point) => {
                  counts(centerIndex) += 1
                  costAccum.add(cost)
                  ZFBLAS.axpy(1.0, point._1, sums(centerIndex))
                  point._2(0) = centerIndex
                  point._2(1) = cost
                }
              }
            } else {
              cenCostAB(i) match {
                case (_, _, _, point) => {
                  val oldCenterIndex = point._2(0).toInt
                  val oldCost = point._2(1)
                  counts(oldCenterIndex) += 1
                  costAccum.add(oldCost)
                  ZFBLAS.axpy(1.0, point._1, sums(oldCenterIndex))
                }
              }
            }
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

  def runAlgorithm(data: RDD[Vector], origCenters: Array[Vector]): Array[Vector] = {

    if (data.getStorageLevel == StorageLevel.NONE) {
      System.err.println("------------------The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.------------------------")
    }

    val sc = data.sparkContext
    val centers = origCenters.clone() //data.takeSample(false, k)
    for (it <- 0 until itN) {
      val bcCenters = sc.broadcast(centers)
      val costAccum = sc.doubleAccumulator

      val totalContibs = data.mapPartitions { points =>
        val jedis = new Jedis("localhost")
        val nnMapT = System.currentTimeMillis()
        val thisCenters = bcCenters.value
        val dims = thisCenters(0).size
        val sums = Array.fill(k)(Vectors.zeros(dims))
        val counts = Array.fill(k)(0L)
        points.foreach { point =>
          val (centerIndex, cost) = ZFKmeansPart.zfFindClosest(point, thisCenters)
          counts(centerIndex) += 1
          ZFBLAS.axpy(1.0, point, sums(centerIndex))
          costAccum.add(cost)
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
    } //end-it
    centers
  }

  def zfComputeCost(data: RDD[Vector], centers: Array[Vector]): Double = {
    val ans = data.map(point => ZFKmeansPart.zfFindClosest(point, centers)._2).sum()
    ans
  }

}


object ZFKmeansPart {
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
    //    testKMeans(sc)
    val k = args(0).toInt //2
    val itN = args(1).toInt //5
    val numFeatures = args(2).toInt //102660
    val centerPath = args(3)
    val dataPath = args(4) //"/Users/zhangfan/Documents/data/kmeans_data.txt"
    val test100 = args(5).toBoolean
    val doGradient = args(6).toBoolean
    val isSparse = args(7).toBoolean
    val minPartN = args(8).toInt


    val rDivN = 100
    val ratioL = if (test100) List(rDivN) else List(35, 36, 37, 38, 39, 40)
    //    val ratioL = if (test100) List(rDivN) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60)
    val mesb = new ArrayBuffer[Double]()
    val nntimesb = new ArrayBuffer[Double]()
    for (r <- ratioL) {
      val ratio = r / rDivN.toDouble
      val jedis = new Jedis("localhost")
      jedis.flushAll()

      val data: RDD[Vector] = if (isSparse) {
        MLUtils.loadLibSVMFile(sc, dataPath, numFeatures, minPartN).map(point => point.features)
      } else {
        sc.textFile(dataPath, minPartN).map(s => Vectors.dense(s.split(",").map(_.toDouble)))
      }
      val train = if (!doGradient) data.sample(false, ratio).cache() else data.cache()
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


      val zfkmeans = new ZFKmeansPart(k, itN)
      val centers = if (doGradient) {
        zfkmeans.runGradientAlgorithm(train, origCenters, ratio)
      } else {
        zfkmeans.runAlgorithm(train, origCenters)
      }

      val partN = jedis.get("partN").split(",")
      val nnMapT = jedis.get("nnMapT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)

      val WSSSE = zfkmeans.zfComputeCost(data, centers)
      mesb += WSSSE
      nntimesb += nnMapT.sum

      println()
      println("dataPart," + data.getNumPartitions + ",trainPart," + train.getNumPartitions + ",nnMapT," + nnMapT.sum + ",origCenter," + origCenters.map(vec => vec.apply(0)).slice(0, 5).mkString(","))
      //      println("partN," + partN.slice(0, math.min(partN.length, 20)).mkString(","))
      println(",ratio," + ratio + ",k," + k + ",itN," + itN + ",doGradient," + doGradient + ",trainN," + train.count() / 10000.0 + ",testN," + data.count() / 10000.0 + ",numFeatures," + train.first().size)
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
