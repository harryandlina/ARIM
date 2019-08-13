package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.apache.spark.mllib.regression.LabeledPoint

class LabeledPointToSample() extends Transformer[LabeledPoint,Sample[Float]] {
  private val featureBuffer = Tensor[Float]()
  private val labelBuffer = Tensor[Float](1)
  private val featureSize = new Array[Int](1)

  override def apply(prev: Iterator[LabeledPoint]): Iterator[Sample[Float]] = {
    prev.map(lp => {
      labelBuffer.storage().array()(0) = lp.label.toFloat
      featureSize(0) = lp.features.size
      featureBuffer.set(Storage(lp.features.toArray.map(_.toFloat)),sizes = featureSize)
      Sample(featureBuffer,labelBuffer)
    })
  }
}

object LabeledPointToSample{
  def apply(): LabeledPointToSample = new LabeledPointToSample()
}