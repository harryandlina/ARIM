package AccurateML.svm.shun.rank

/**
  * ICT
  * Created by douyishun on 11/30/16.
  */

import java.io._

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression._
import org.apache.spark.rdd.RDD


/**
  * Create SVM model.
  * @param svIndexedData Support Vector point.
  * @param svAlpha The alpha corresponding to Support Vector.
  * @param intercept Intercept or called bias.
  * @param kernelFunc kernel function.
  */
class SVMModel(svIndexedData: Array[LabeledPoint],
               svAlpha: Array[Double],
               intercept: Double,
               kernelFunc: (Vector, Vector) => Double) extends java.io.Serializable{

  def predictPoint(x: Vector): Int = {
    var score: Double = -1 * intercept
    for(i <- svAlpha.indices) {
      score += svAlpha(i) * svIndexedData(i).label * kernelFunc(svIndexedData(i).features, x)
    }
    if(score > 0) 1
    else -1
  }

  def modelAccuracy(scoreAndLabels: RDD[(Int, Int)]): Double ={
    scoreAndLabels.map(
      x =>
        if (x._1 == x._2) 1
        else 0
    ).reduce((a,b) => a + b).toDouble / scoreAndLabels.count()
  }

  def printParameter(fileName: String): Unit ={
    printToFile(new File(fileName)){p =>
      svAlpha.foreach(p.println)
    }
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }
}
