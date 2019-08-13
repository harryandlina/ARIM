package AccurateML.blas

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangfan on 17/2/15.
  */
class Cancel {

}

object Cancel {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)

    val srcPath = args(0)
    val desPath = args(1)
    val data = sc.textFile(srcPath) //"hdfs:/cc.txt"
    //    println(data.count())
    data.saveAsTextFile(desPath)
    println(data.count() + "\tdone")

  }
}
