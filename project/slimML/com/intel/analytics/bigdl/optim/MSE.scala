package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{AbsCriterion, ClassNLLCriterion, MSECriterion}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.commons.lang3.SerializationUtils

import scala.reflect.ClassTag

/**
  * This evaluation method is calculate mean squared error of output with respect to target
  *
  */
class MSE[@specialized(Float, Double)T: ClassTag]()
                                                 (implicit ev: TensorNumeric[T]) extends ValidationMethod[T] {
  private val criterion = MSECriterion[T]()
  override def apply(output: Activity, target: Activity): LossResult = {
    val _output = output.asInstanceOf[Tensor[T]]
    val (max_prob, max_index) = _output.max(2)
    val _target = target.asInstanceOf[Tensor[T]]
    val loss = ev.toType[Float](criterion.forward(max_index, _target))
    //    val loss = ev.toType[Float](criterion.forward(max_prob, _target))
    val count = 1

    new LossResult(loss, count)
  }

  override def format(): String = "MSE"
}