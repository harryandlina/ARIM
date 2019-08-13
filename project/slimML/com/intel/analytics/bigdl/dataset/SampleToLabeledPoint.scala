package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

class SampleToLabeledPoint() extends Transformer[Sample[Float],LabeledPoint] {
  private val featureBuffer = Tensor[Float]()
  private val labelBuffer = Tensor[Float]()
  private val featureSize = new Array[Int](1)
  override def apply(prev: Iterator[Sample[Float]]): Iterator[LabeledPoint] = {
    prev.map(sample => {
      featureSize(0) = sample.feature().nElement()
      require(sample.label().nElement() == 1,s"the label of samples should equle to 1,found ${sample.label().size().mkString("x")}")
      labelBuffer.copy(sample.label())
      featureBuffer.copy(sample.feature())
      val features = featureBuffer.reshape(featureSize).toArray().map(_.toDouble)
      LabeledPoint(labelBuffer.valueAt(0).toDouble,Vectors.dense(features))
    })
  }
}

object SampleToLabeledPoint{
  def apply(): SampleToLabeledPoint = new SampleToLabeledPoint()
}