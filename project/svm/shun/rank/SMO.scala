package AccurateML.svm.shun.rank

/**
  * ICT
  * Created by douyishun on 11/23/16.
  */

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression._

import scala.collection.{Map, mutable}

/**
  * The algorithm for training SVM based on Platt's SMO and two
  * threshold parameters are employed to derive modification.
  *
  * @param initAlpha Initial alpha.
  * @param maxIterations Max iteration of takeStep(x1, x2).
  * @param C Penalty factor.
  * @param tol Tolerance factor for checking violation and convergence.
  * @param eps Permissible error for checking boundary conditions.
  * @param kernelFunc Kernel function.
  */
class SMO(initAlpha: Map[Long, Double],
          maxIterations: Int,
          C: Double,
          tol: Double,
          eps: Double,
          kernelFunc: (Vector, Vector) => Double) extends java.io.Serializable {

  private val indexedPoint = new mutable.HashMap[Long, Vector]
  private val y = new mutable.HashMap[Long, Int]//label
  private val fCache = new mutable.HashMap[Long, Double]//init fCache with -y_i
  private val alpha = new mutable.HashMap[Long, Double]
  private var indices = Array[Int]()
  private var n: Long = 0//total num of point in the partition
  private var b_low: Double = 1; var b_up: Double = -1
  private var i_low: Int = -1; var i_up: Int = -1
  private var iteration = 0

  /**
    * Train a sub-classifier for current partition.
 *
    * @param partitionIndex The index of current partition.
    * @param iter  The iterator of current indexedPoint.
    * @return Iterator of alpha and intercept((b_low+b_up)/2).
    */
  def train(partitionIndex: Int,
            iter: Iterator[(Long, LabeledPoint)]): Iterator[(Long, Double)] = {

    var tmpIndices = mutable.ArrayBuffer[Int]()
    while(iter.hasNext) {
      val cur = iter.next()
      tmpIndices += cur._1.toInt
      indexedPoint += (cur._1 -> cur._2.features)
      y += (cur._1 -> cur._2.label.toInt)
      fCache += (cur._1 -> -1 * cur._2.label)
      alpha += (cur._1 -> (if(initAlpha.contains(cur._1)) initAlpha(cur._1) else 0.0))
    }
    indices = tmpIndices.toArray
    n = indices.length

    val kernelMatrix = Kernel.computeKernelMatrix(indexedPoint, kernelFunc)

    // init i_low to any one index of class_2(y_i == -1)
    //       i_up to any one index of class_1(y_i == 1)
    for(i <- indices) {
      if(y(i) == -1) i_low = i
      else i_up = i
    }

    //training begin
    var numChanged: Int = 0
    var examineAll: Boolean = false
    while((numChanged > 0 || !examineAll) && iteration < maxIterations) {
      numChanged = 0
      if(!examineAll) {//loop for all
        for(i2 <- indices) { numChanged += examineExample(i2, kernelMatrix) }
      }
      else {//loop for I_0
        var inner_loop_success: Int = 1
        while(b_low > b_up + 2 * tol &&
              inner_loop_success == 1 &&
              iteration < maxIterations) {
          inner_loop_success = takeStep(i_up, i_low, kernelMatrix)
          numChanged += inner_loop_success
        }
        numChanged = 0
      }

      //loop: All -> I_0 -> All -> ... Until convergent for all point or up to max iteration.
      if(!examineAll) examineAll = true
      else if(numChanged == 0) examineAll = false
    }
    //training end

    //Add intercept to alpha with ((-1 * partitionIndex - 1) -> intercept))
    alpha += ((-1L * partitionIndex - 1) -> ((b_low + b_up) / 2))
    alpha.iterator
  }

  /**
    * Examine whether the point with index of i2 should be optimized or not.
 *
    * @return "1": Successful optimization(takeStep()).
    *         "0": Without optimizing or failing optimization(takeStep()).
    */
  private def examineExample(i2: Int, kernelMatrix: Map[(Long, Long), Double]): Int = {
    var i1: Int = 0
    val alpha_i2: Double = alpha(i2)
    var f_i2: Double = fCache(i2)
    if(alpha_i2 < eps || alpha_i2 > C - eps) {//i2 not belongs to I_0
      f_i2 = computeFi(i2, kernelMatrix)//compute f_i2
      fCache(i2) = f_i2

      //update (b_low, i_low), (b_up, i_up)
      if(iBelongsTo(i2) == 1 && f_i2 < b_up) {b_up = f_i2; i_up = i2}
      else if(iBelongsTo(i2) == -1 && f_i2 > b_low) {b_low = f_i2; i_low = i2}
    }

    //check violation
    var optimality: Int = 1
    if(iBelongsTo(i2) == 1) {
      if(b_low - f_i2 > 2 * tol) {optimality = 0; i1 = i_low}
    }
    if(iBelongsTo(i2) == -1) {
      if(f_i2 - b_up > 2 * tol) {optimality = 0; i1 = i_up}
    }
    if(optimality == 1) return 0

    //for i2 in I_0 choose the better i1
    if(alpha_i2 > eps && alpha_i2 < C - eps) {
      if(b_low - f_i2 > f_i2 - b_up) i1 = i_low
      else i1 = i_up
    }
    if(takeStep(i1, i2, kernelMatrix) == 1) 1
    else 0
  }


  /**
    * Take step(optimize) for i1 and i2.
    * Update alpha, FCache, (b_low, i_low) and (b_up, i_up).
 *
    * @return "1": Successful optimization.
    *         "0": Failing optimization.
    */
  private def takeStep(i1: Int, i2: Int,
                       kernelMatrix: Map[(Long, Long), Double]): Int = {
    if(i1 == i2 || i1 < indices(0) || i1 >= indices(0) + n || i2 < indices(0) || i2 >= indices(0) + n)
      return 0
    val sign: Int = y(i1) * y(i2)
    val alphaOld_i1 = alpha(i1); val alphaOld_i2 = alpha(i2)
    var alphaNew_i1: Double = 0; var alphaNew_i2: Double = 0
    var L: Double = 0; var H: Double = 0

    //compute L, H
    if (sign < 0) {
      L = math.max(0, alphaOld_i2 - alphaOld_i1)
      H = math.min(C, C + alphaOld_i2 - alphaOld_i1)
    }
    else {
      L = math.max(0, alphaOld_i2 + alphaOld_i1 - C)
      H = math.min(C, alphaOld_i2 + alphaOld_i1)
    }
    if(L == H) return 0

    val eta: Double = 2 * kernelMatrix((i1, i2))
                        - kernelMatrix((i1, i1))
                        - kernelMatrix((i2, i2))
    if(eta < 0) {
      alphaNew_i2 = alphaOld_i2 - y(i2) * (fCache(i1) - fCache(i2)) / eta
      if(alphaNew_i2 < L) alphaNew_i2 = L
      else if(alphaNew_i2 > H) alphaNew_i2 = H
    }
    else {
      //Lobj - Hobj = y2(F1*L - F1*H - F2*L + F2*H)
      val diff = y(i2) * (fCache(i1) - fCache(i2)) * (L - H)
      if(diff > eps) alphaNew_i2 = L
      else if(diff < -1 * eps) alphaNew_i2 = H
      else alphaNew_i2 = alphaOld_i2
    }

    //move too short
    if(math.abs(alphaNew_i2 - alphaOld_i2) < eps * (alphaNew_i2 + alphaOld_i2 + eps))
      return 0

    alphaNew_i1 = alphaOld_i1 + sign * (alphaOld_i2 - alphaNew_i2)
    alpha(i1) = alphaNew_i1; alpha(i2) = alphaNew_i2

    //update fCache
    updateFCache(i1, i2, alphaNew_i1, alphaNew_i2, alphaOld_i1, alphaOld_i2, kernelMatrix)

    //update (b_low, i_low), (b_up, i_up) using i1, i2 and indices in I_0
    val (tmp1, tmp2) = findI_lowI_up(i1, i2)
    i_low = tmp1; i_up = tmp2
    b_low = fCache(i_low); b_up = fCache(i_up)

    iteration += 1

    1
  }

  /**
    * Find i_low and i_up only based on i1, i2 and indices in I_0
 *
    * @return (max_i, min_i): corresponding to (i_low, i_up)
    */
  private def findI_lowI_up(i1: Int, i2: Int): (Int, Int) = {
    var min: Double = 0x3f3f3f3f; var max: Double = -min
    var min_i: Int = -1; var max_i: Int = -1
    for(i <- indices) {
      if((alpha(i) > eps && alpha(i) < C - eps) || i == i1 || i == i2) {
        if(fCache(i) > max) { max = fCache(i); max_i = i }
        if(fCache(i) < min) { min = fCache(i); min_i = i }
      }
    }
    (max_i, min_i)
  }


  private def computeFi(i2: Int, kernelMatrix: Map[(Long, Long), Double]): Double = {
    var ans: Double = -1 * y(i2)
    for(i <- indices) {
      ans += alpha(i) * y(i) * kernelMatrix((i,i2))
    }
    ans
  }

  /**
    * Update FCache for i1, i2 and indices in I_0.
    */
  private def updateFCache(i1: Int, i2:Int,
                           alphaNew_i1: Double, alphaNew_i2: Double,
                           alphaOld_i1: Double, alphaOld_i2: Double,
                           kernelMatrix: Map[(Long, Long), Double]): Unit = {
    for(i <- indices) {
      if((alpha(i) > eps && alpha(i) < C - eps) || i == i1 || i == i2)
        fCache(i) = fCache(i) + y(i1) * (alphaNew_i1 - alphaOld_i1) * kernelMatrix(i1, i)
                              + y(i2) * (alphaNew_i2 - alphaOld_i2) * kernelMatrix(i2, i)
    }
  }

  /**
    * Find the classes which i belongs to.
 *
    * @return "-1" for {i: i in {I_0 & I_3 & I_4}}
    *         "1"  for {i: i in {I_0 & I_1 & I_2}}
    */
  private def iBelongsTo(i: Int): Int = {
    var ans: Int = 0

    //{i: i in {I_0 & I_3 & I_4}}
    val alpha_i = alpha(i)
    if((y(i) == 1 && alpha_i > C - eps) ||
      (y(i) == -1 && alpha_i < eps) ||
      (alpha_i > eps && alpha_i < C - eps)) {ans = -1}

    //{i: i in {I_0 & I_1 & I_2}}
    if((y(i) == 1 && alpha_i < eps) ||
      (y(i) == -1 && alpha_i > C - eps) ||
      (alpha_i > eps && alpha_i < C - eps)) {ans = 1}
    ans
  }
}