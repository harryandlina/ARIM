package AccurateML.blas

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

/**
  * Created by zhangfan on 16/10/17.
  */
object ZFUtils {
  def zfChangeFSize(point: LabeledPoint, m: Int): LabeledPoint = {
    val ff = point.features.toSparse
    new LabeledPoint(point.label, Vectors.sparse(m, ff.indices, ff.values))
  }

  def powerExponent(data: LabeledPoint, power: Int): LabeledPoint = {
    if (power == 1) {
      data
    } else {
      data.features match {
        case dd: SparseVector => {
          val datai = data.features.toSparse.indices
          val datav = data.features.toSparse.values
          val sm = datai.size
          val index = new Array[Int](sm * power)
          val values = new Array[Double](sm * power)
          for (i <- 0 until sm) {
            for (j <- 0 until power) {
              val newi = i * power + j
              index(newi) = datai(i) * power + j
              values(newi) = math.pow(datav(i), j + 1)
            }
          }
          var newlabel = 0.0
          for (fv <- values)
            newlabel += fv //weight = 1
          new LabeledPoint(newlabel, Vectors.sparse(data.features.size * power, index, values))
          //          new LabeledPoint(data.label, Vectors.sparse(data.features.size*power,index,values))
        }
        case ds: DenseVector => {
          val features = data.features.toArray
          val m = features.size
          val newfeatures = new Array[Double](m * power)
          for (i <- features.indices) {
            val base = features(i)
            for (j <- 0 until power) {
              newfeatures(i * power + j) = math.pow(base, j + 1)
            }
          }
          var newlabel = 0.0
          for (fv <- newfeatures)
            newlabel += fv
          new LabeledPoint(newlabel, Vectors.dense(newfeatures))
          //          new LabeledPoint(data.label, Vectors.dense(newfeatures))
        }
      }
    }
  }

  def zfParseDouble(s: String) = try {
    Some(s.toDouble)
  } catch {
    case _: Throwable => None
  }

  def zfnorm(vec: Vector): Double = {
    var sum = 0.0
    for (v <- vec.toArray) {
      sum += v * v
    }
    math.sqrt(sum)
  }

  def zfCosin(a: Vector, b: Vector): Double = {
    ZFBLAS.dot(a, b) / (Vectors.norm(a, 2) * Vectors.norm(b, 2))
  }

  def zfEuclideanDistance(a: Vector, b: Vector): Double = {
    val temp = b.copy
    ZFBLAS.axpy(-1, a, temp)
    Vectors.norm(temp, 2)
  }
  def zfmean(xs: Array[Double]): Double = xs match {
    case ys => ys.reduceLeft(_ + _) / ys.size.toDouble
  }
  def zfstddev(xs: Array[Double], avg: Double): Double = xs match {
    case ys => math.sqrt((0.0 /: ys) {
      (a,e) => a + math.pow(e - avg, 2.0)
    } / xs.size)
  }

  def zf2Float(d: Double): String = {
    f"$d%.2f"
  }

  //  def zfQuitSort(a:Array[Double],left:Int,right:Int): Unit ={
  //    var i=left
  //    var j=right
  //    var temp=0.0
  //    val pivot=a((left+right)/2)
  //    while (i<=j){
  //      while (a(i)<pivot)
  //        i+=1
  //      while (a(j)>pivot)
  //        j-=1
  //      if(i<=j){
  //        temp=a(i)
  //        a(i)=a(j)
  //        a(j)=temp
  //        i+=1
  //        j-=1
  //      }
  //    }
  //    if(left<j)
  //      zfQuitSort(a,left,j)
  //    if(i<right)
  //      zfQuitSort(a,i,right)
  //  }
  def zfQuitSort(a: Array[Double], left: Int, right: Int): Unit = {
    var j = right
    var i = left
    val pos = j
    j -= 1
    var dowhile: Boolean = true
    while (i <= j && dowhile) {
      while (i < pos && a(i) <= a(pos))
        i += 1
      while (j >= 0 && a(j) > a(pos))
        j -= 1
      if (i >= j) {
        dowhile = false
      } else {
        val temp = a(i)
        a(i) = a(j)
        a(j) = temp
      }
    }
    val temp = a(i)
    a(i) = a(pos)
    a(pos) = temp
    i += 1
    if (left < j)
      zfQuitSort(a, left, j)
    if (i < right)
      zfQuitSort(a, i, right)
  }

  def partition(a: Array[Double], l: Int, r: Int): Int = {
    var right = r
    var left = l
    val pos = right
    right -= 1
    var dowhile: Boolean = true
    while (left <= right && dowhile) {
      while (left < pos && a(left) <= a(pos))
        left += 1
      while (right >= 0 && a(right) > a(pos))
        right -= 1
      if (left >= right) {
        dowhile = false
      } else {
        val temp = a(left)
        a(left) = a(right)
        a(right) = temp
      }
    }
    val temp = a(left)
    a(left) = a(pos)
    a(pos) = temp
    left
  }

  def zfFindMeadian(a: Array[Double]): Double = {
    var left = 0
    var right = a.size - 1
    val midPos = right / 2
    var index = -1
    var dowhile: Boolean = true
    while (index != midPos && dowhile) {
      index = partition(a, left, right)
      if (index < midPos)
        left = index + 1
      else if (index > midPos)
        right = index - 1
      else
        dowhile = false
    }
    assert(index == midPos)
    a(index)
  }

  def main(args: Array[String]) {
    val a = Array[Double](0.0) //1.0,2.0,113.0,-1.0,112.0,7.0,4.0
    val b = a.clone()
    zfQuitSort(b, 0, a.size - 1)
    println(a.mkString(","))
    println(b.mkString(","))

    println(zfFindMeadian(a.clone()) + "," + b((b.size - 1) / 2))


  }

}
