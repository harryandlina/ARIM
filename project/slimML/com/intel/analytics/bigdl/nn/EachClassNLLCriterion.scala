package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.SizeAverageStatus.SizeAverageStatus
import com.intel.analytics.bigdl.nn.abstractnn.{SizeAverageStatus, TensorCriterion}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.utils.Engine
import org.apache.hadoop.mapreduce.v2.app.speculate.TaskRuntimeEstimator

class EachClassNLLCriterion[@specialized(Float, Double) T: ClassTag]
(weights: Tensor[T] = null, sizeAverage: Boolean = true,
 logProbAsInput: Boolean = true, paddingValue: Int = -1)
(implicit ev: TensorNumeric[T]) extends Serializable {
  var total_weight = ev.fromType[Int](0)
  if (weights != null) require(weights.dim() == 1,
    "weights input should be 1-D Tensor" +
      s"weights dim(${weights.dim()})")

  @transient
  private var results: Array[Future[(T, T)]] = null
  @transient
  private var resultsBackward: Array[Future[_]] = null

  private val epsilon: T = ev.fromType(1e-8)

  private val oneMinusEpsilon: T = ev.minus(ev.one, epsilon)

  var output: Array[T] = null

  /**
    * Takes an input object, and computes the corresponding loss of the criterion,
    * compared with `target`.
    *
    * @param input input data
    * @param target target
    * @return the loss of criterion
    */
  def forward(input: Tensor[T], target: Tensor[T]): Array[T] = {
    updateOutput(input, target)
  }

  def updateOutput(input: Tensor[T], target: Tensor[T]): Array[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "ClassNLLCriterion: " +
        ErrorInfo.constrainInputAsVectorOrBatch +
        s"input dim(${input.dim()})")
    val nClasses = input.size(input.dim())
    if (input.dim() == 1) {
      require(input.dim() == target.dim(),
        "ClassNLLCriterion: " + ErrorInfo.constrainInputDimSameAsTarget +
          s" Input dimension is: ${ input.dim() } , target dimension is: ${ target.dim() }")
      val curTarget = ev.toType[Int](target.valueAt(1))
      assert(curTarget >= 1 && curTarget <= nClasses || curTarget == paddingValue,
        s"curTarget ${curTarget} is out of range, should be 1 to ${nClasses}")
      total_weight = if (weights != null) weights(Array(curTarget)) else ev.fromType[Int](1)
      output = if (curTarget == paddingValue) Array(ev.zero)
      else {
        if (!logProbAsInput) {
          val clipped = ev.clip(input.valueAt(curTarget), epsilon, oneMinusEpsilon)
          Array(ev.times(ev.negative(ev.log(clipped)), total_weight))
        } else {
          Array(ev.times(ev.negative(input.valueAt(curTarget)), total_weight))
        }
      }
    } else if (input.dim() == 2) {
      val batchSize = input.size(1)
      val targetSize = target.size()
      target.squeeze()
      require(target.dim() == 1,
        "ClassNLLCriterion: illegal target! Target should be 1D tensor after squeeze," +
          s"but target's size is: ${ target.size() }, please check your data.")

      total_weight = ev.fromType[Int](0)
      output = new Array[T](batchSize)

      if (results == null || results.length != batchSize) {
        results = new Array[Future[(T, T)]](batchSize)
      }

      var i = 1
      while (i <= batchSize) {
        val _i = i
        results(_i - 1) = Engine.model.invoke( () => {
          val curTarget = ev.toType[Int](target.valueAt(_i))
          assert(curTarget >= 1 && curTarget <= nClasses || curTarget == paddingValue,
            s"curTarget ${curTarget} is out of range 1 to ${nClasses}")
          if (curTarget == paddingValue) (ev.zero, ev.zero)
          else {
            val curWeight = if (weights != null) weights.valueAt(curTarget) else ev.fromType[Int](1)
            if (!logProbAsInput) {
              val clipped = ev.clip(input.valueAt(_i, curTarget), epsilon, oneMinusEpsilon)
              (ev.times(ev.log(clipped), curWeight), curWeight)
            } else {
              (ev.times(input.valueAt(_i, curTarget), curWeight), curWeight)
            }

          }
        })
        i += 1
      }

      i = 0
      while (i < batchSize) {
        val (o, w) = Await.result(results(i), Duration.Inf)
        output(i) = ev.negative(o)
        total_weight = ev.plus(total_weight, w)
        i += 1
      }
      if (total_weight == 0) {
        total_weight = ev.fromType[Int](1)
      }
      target.resize(targetSize)
    }
    output
  }
}

object EachClassNLLCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      weights: Tensor[T] = null,
                                                      sizeAverage: Boolean = true,
                                                      logProbAsInput: Boolean = true,
                                                      paddingValue: Int = -1
                                                    )(implicit ev: TensorNumeric[T]) : EachClassNLLCriterion[T] = {
    new EachClassNLLCriterion[T](weights, sizeAverage, logProbAsInput, paddingValue)
  }
}
