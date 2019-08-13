package slpart.models.lenet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

object LeNet5 {
  def apply(classNum: Int = 10): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv_1")) //(28-5+2*0)/1 +1 = 24    24x24x6
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2).setName("pool_1"))//(24-2+2*0)/2 + 1 = 12     12x12x6
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv_2"))//(12-5+2*0)/1  +1 = 8    8x8x12
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2).setName("pool_2")) //(8-2+2*0)/2 +1 = 4     4x4x12
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc_3"))
      .add(Tanh())
      .add(Linear(100, classNum).setName("fc_4"))
      .add(LogSoftMax())
    model
  }
}
