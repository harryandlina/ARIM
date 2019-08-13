package AccurateML.blas

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by zhangfan on 16/11/8.
  */
object ZFCaculateEuclideanMatrix {
  def main(args: Array[String]) {
    val fileName="/Users/zhangfan/Downloads/cancel"
    val iter = Source.fromFile(fileName).getLines()
    var i=1
    var lines = new ArrayBuffer[String]()
    while(iter.hasNext){
      lines += iter.next()
      i+=1
    }
    val vecs = lines.toArray.map(line=>Vectors.dense(line.split(",").map(_.toDouble)))
    val n=lines.length
    val matrix = Array.ofDim[Double](n,n)
    for(i<-0 until n){
      for(j<-0 until i){
        matrix(i)(j) = ZFUtils.zfEuclideanDistance(vecs(i),vecs(j))
        matrix(j)(i)=matrix(i)(j)
      }
    }
    for(i<-0 until n){
      println(matrix(i).mkString(","))
    }
  }

}
