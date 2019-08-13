package com.intel.analytics.bigdl.optim

import org.apache.spark.AccumulatorParam

object ArrayAccumulator extends AccumulatorParam[Array[Double]]{
  override def zero(initialValue: Array[Double]): Array[Double] = {
    Array()
  }

  override def addInPlace(r1: Array[Double], r2: Array[Double]): Array[Double] = {
    r1 ++ r2
  }

}
