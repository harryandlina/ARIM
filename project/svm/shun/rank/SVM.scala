package AccurateML.svm.shun.rank

/**
  * ICT
  * Created by douyishun on 12/7/16.
  */

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression._
import org.apache.spark.rdd.RDD

import scala.collection.mutable

/**
  * Train SVM Model by multi-layer SVMs. Each layer contains multiple parallel SMOs.
  * The middle results between layers will filter the training example and generate the
  * initial parameter of next layer.
  *
  * At each layer, the partial results(alpha and bias) from parallel SMOs are combined
  * and used for filtering the point which may cause great impact to the model. And the
  * filtered point is the training example of next layer, significantly reduce training
  * size.
  */
class SVM(trainingData: RDD[LabeledPoint]) extends java.io.Serializable {

  def train(zfratio: Double,
            maxLayer: Int,
            maxIterations: Int,
            C: Double,
            tol: Double,
            eps: Double,
            kernelFunc: (Vector, Vector) => Double): SVMModel = {

    val scaledData = Scale.scale(trainingData)
    val indexedData = scaledData.zipWithIndex().map { case (v, k) => (k, v) }.cache()
    val n: Int = indexedData.count().toInt //total number of training data
    val indexedDataMap = indexedData.collectAsMap()

    var topIndexedData: RDD[(Long, LabeledPoint)] = indexedData //indexedData.sample(false, zfratio, 47L)
    val alpha = new mutable.HashMap[Long, Double]
    for (i <- 0 until n) alpha += (i.toLong -> 0.0) //init alpha with 0
    var intercept: Double = 0

    //training begin

    for (i <- 1 to maxLayer) {
      val smo = new SMO(alpha, maxIterations, C, tol, eps, kernelFunc)
      val res = topIndexedData.mapPartitionsWithIndex(smo.train)
      res.persist()

      //update alpha and intercept
      val updatedAlpha = res.filter(x => x._1 == 0 || x._1 > 0).collectAsMap()
      for (i <- updatedAlpha.keys) alpha(i) = updatedAlpha(i)
      intercept = res.filter(x => x._1 < 0).map(x => x._2).mean()
      res.unpersist()

      //      select most important k%
      var score: Double = 0
      val topIndex = indexedData.map(
        x => {
          score = -1 * intercept
          for (j <- alpha.keys if alpha(j) > 0) {
            score += alpha(j) * indexedDataMap(j).label *
              kernelFunc(indexedDataMap(j).features, indexedDataMap(x._1).features)
          }
          (Math.abs(indexedDataMap(x._1).label * score - 1), x._1.toInt)
        }
      ).top((n * zfratio).toInt)
      topIndexedData = indexedData.filter(x => topIndex.map(_._2).contains(x._1))

    }
    //training end

    //collect support vector
    val svIndex: Array[Long] = alpha.filter(x => x._2 > 0).keys.toArray
    val svIndexedData = indexedDataMap.filter(x => svIndex.contains(x._1))
    val svAlpha = alpha.filter(x => svIndex.contains(x._1))

    //build SVMModel with support vector
    new SVMModel(svIndexedData.values.toArray,
      svAlpha.values.toArray, intercept, kernelFunc)
  }

  def train(zfratio: Double,
            maxLayer: Int,
            maxIterations: Int,
            C: Double,
            gamma: Double): SVMModel = {
    val kernel = new Kernel
    train(zfratio, maxLayer, maxIterations, C, 1e-3, 1e-3, kernel.rbf(gamma))
  }
}
