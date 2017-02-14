package AccurateML.svm.shun.hash

/**
  * ICT
  * Created by douyishun on 11/23/16.
  */

import java.io.FileWriter

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis
import AccurateML.blas.ZFUtils
import AccurateML.nonLinearRegression.ZFHash3

object ZipSize {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("SVMKernel")
    val sc = new SparkContext(conf)

    val initW: Double = args(0).toDouble
    val step: Double = args(1).toDouble
    val numFeature: Int = args(2).toInt //10
    val ineritN: Int = args(3).toInt //5
    val itN: Int = args(4).toInt
    val testPath: String = args(5)
    val dataPath: String = args(6)
    val test100: Boolean = args(7).toBoolean
    val weightsPath = args(8)
    val minPartN = args(9).toInt

    val itqbitN = args(10).toInt
    val itqitN = args(11).toInt
    val itqratioN = args(12).toInt //from 1 not 0
    val upBound = args(13).toInt
    val splitN = args(14).toDouble

    val data = MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN) //"data/" + dataSet + "/train"
    val test = MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN) //"data/" + dataSet + "/test"

    val jedis = new Jedis("localhost")
    jedis.flushAll()
    val oHash = new ZFHash3(itqbitN, itqitN, itqratioN, upBound, splitN, true)
    val objectData = data.mapPartitions(oHash.zfHashMap).map(_._1).persist(StorageLevel.MEMORY_AND_DISK)

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

    val zips:RDD[LabeledPoint]=objectData.flatMap(o=>o(0))
    val originals:RDD[LabeledPoint] = objectData.flatMap(o=>o(1))

    val zipN = zips.count()
    val origN = originals.count()

    println("zipN,"+zipN+",originalN,"+origN+",zipR,"+zipN/origN.toDouble)
    zips.repartition(1).saveAsObjectFile(dataPath+".zip")
    originals.repartition(1).saveAsObjectFile(dataPath+".original")



  }
}
