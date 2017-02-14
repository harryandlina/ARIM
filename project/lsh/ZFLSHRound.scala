package AccurateML.lsh

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import redis.clients.jedis.Jedis
import AccurateML.blas.ZFBLAS

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

/**
  * Created by zhangfan on 16/9/21.
  */
class ZFLSHRound(
                  lshRound: Int,
                  lshPerRound: Int,
                  lshPerBucketN: Int
                ) extends Serializable {


  def zfMapFuncLSHRound(pit: Iterator[LabeledPoint]): Iterator[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])] = {
    if(pit.isEmpty){
      val ansAll = new ArrayBuffer[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])]()
      ansAll.iterator
    }else{
      val jedis = new Jedis("localhost")
      var time = System.currentTimeMillis()
      val ansAll = new ArrayBuffer[(Array[ArrayBuffer[LabeledPoint]], Array[ArrayBuffer[(Int, Int)]], ArrayBuffer[mutable.HashSet[Int]])]()
      var i = 0
      val train = new collection.mutable.ArrayBuffer[LabeledPoint]()
      while (pit.hasNext) {
        val point = pit.next()
        train += point
        i += 1
      }
      val trainsN = i
      val isDense: Boolean = {
        train.last.features match {
          case v: DenseVector => true
          case v: SparseVector => false
        }
      }
      val numFeatures = train.last.features.size

      val firstSets = new ArrayBuffer[mutable.HashSet[Int]]()
      firstSets ++= lshClusterRound(train, mutable.HashSet(Array.range(0, trainsN): _*), numFeatures, lshPerBucketN, lshPerRound)
      for (aset <- firstSets) {
        val asetTrain = new ArrayBuffer[LabeledPoint]()
        for (index <- aset) {
          asetTrain += train(index)
        }
        val roundNums = new Array[ArrayBuffer[Int]](lshRound)
        val zips = new Array[ArrayBuffer[LabeledPoint]](lshRound + 1)
        val roundIndexs = new Array[ArrayBuffer[(Int, Int)]](lshRound - 1)
        val oldSets = new ArrayBuffer[mutable.HashSet[Int]]() //collection.mutable.HashSet[Int](Array.range(0,cnt): _*)
        val newSets = new ArrayBuffer[mutable.HashSet[Int]]()
        oldSets += mutable.HashSet[Int](Array.range(0, asetTrain.size): _*)
        for (r <- 0 until lshRound - 1) {
          newSets.clear()
          roundNums(r) = new ArrayBuffer[Int]()
          zips(r) = new ArrayBuffer[LabeledPoint]()
          roundIndexs(r) = new ArrayBuffer[(Int, Int)]()
          for (oi <- 0 until oldSets.size) {
            val tempn = oldSets(oi).size
            roundNums(r) += tempn
            if (tempn <= lshPerBucketN) {
              newSets += oldSets(oi)
            } else {
              val newlist = lshClusterRound(asetTrain, oldSets(oi), numFeatures, lshPerBucketN, lshPerRound)
              newSets ++= newlist
            }
          }
          oldSets.clear()
          oldSets ++= newSets
        }
        zips(lshRound - 1) = new ArrayBuffer[LabeledPoint]()
        roundNums(lshRound - 1) = new ArrayBuffer[Int]()
        for (oi <- 0 until oldSets.size) {
          roundNums(lshRound - 1) += oldSets(oi).size
        }


        for (r <- 0 until lshRound - 1) {
          var lastj = 0
          for (num <- roundNums(r)) {
            var sum = 0
            val oldj = lastj
            while (lastj < roundNums(r + 1).size && sum < num) {
              sum += roundNums(r + 1)(lastj)
              lastj += 1
            }
            roundIndexs(r) += Tuple2(oldj, lastj - 1)
          }
        }

        for (i <- 0 until oldSets.size) {
          val zipFeature = Vectors.zeros(numFeatures)
          var zipLabel = 0.0
          for (index <- oldSets(i)) {
            val point = asetTrain(index)
            zipLabel += point.label
            ZFBLAS.axpy(1.0, point.features, zipFeature)
          }
          if (isDense) {
            zips(lshRound - 1) += new LabeledPoint(zipLabel, zipFeature.toDense)
          } else {
            zips(lshRound - 1) += new LabeledPoint(zipLabel, zipFeature.toSparse)
          }
        }



        var r = lshRound - 2
        while (r >= 0) {
          for (t <- roundIndexs(r)) {
            val zipFeature = Vectors.zeros(numFeatures)
            var zipLabel = 0.0
            for (zi <- t._1 to t._2) {
              val point = zips(r + 1)(zi)
              zipLabel += point.label
              ZFBLAS.axpy(1.0, point.features, zipFeature)
            }
            if (isDense) {
              zips(r) += new LabeledPoint(zipLabel, zipFeature.toDense)
            } else {
              zips(r) += new LabeledPoint(zipLabel, zipFeature.toSparse)
            }
          }
          r -= 1
        }
        for (r <- 0 to lshRound - 1) {
          for (j <- 0 until zips(r).size) {
            val point = zips(r)(j)
            val n = roundNums(r)(j)
            ZFBLAS.scal(1.0 / n, point.features)
            val azip = {
              if (isDense) new LabeledPoint(point.label / n, point.features.toDense) else new LabeledPoint(point.label / n, point.features.toSparse)
            }
            zips(r)(j) = azip
          }
        }

        zips(lshRound) = asetTrain
        val ans = Tuple3(zips, roundIndexs, oldSets)
        ansAll += ans

        jedis.append("dataN",","+aset.size)

      }
      jedis.append("firstPBN", "," + firstSets.map(_.size).mkString(","))
      time = System.currentTimeMillis() - time
      jedis.append("lshT", "," + time)
      //    for(a<-ansAll){
      //      jedis.append("nn",","+a._1.last.size)
      //    }
      //    jedis.append("on",","+1)
      jedis.close()

      ansAll.iterator
    }

  }

  //  def lshCluster(
  //                  train: mutable.ArrayBuffer[LabeledPoint],
  //                  iset: mutable.HashSet[Int],
  //                  numBits: Int,
  //                  numFeatures: Int
  //                ): List[mutable.HashSet[Int]] = {
  //    val lsh = new ZFLSH(numBits, numFeatures)
  //    //    val key: String = lsh.hashVector(Vectors.dense(Array[Double](1, 1)))
  //    val amap = new mutable.HashMap[String, mutable.HashSet[Int]]()
  //    for (index <- iset) {
  //      val key = lsh.hashVector(train(index).features)
  //      val aset = amap.getOrElse(key, new mutable.HashSet[Int]())
  //      aset += index
  //      amap.update(key, aset)
  //    }
  //    amap.values.toList
  //  }


  //  def lshClusterRound(
  //                       train: mutable.ArrayBuffer[LabeledPoint],
  //                       iset: mutable.HashSet[Int],
  //                       numBits: Int,
  //                       numFeatures: Int,
  //                       perBucketN: Int,
  //                       setIterN: Int,
  //                       itN: Int
  //                     ): List[mutable.HashSet[Int]] = {
  //    if (itN >= setIterN || iset.size <= perBucketN) {
  //      List(iset)
  //    } else {
  //      val bN = iset.size/perBucketN
  //      val bitN = math.log(bN)/math.log(2)+1
  //      val lsh = new ZFLSH(bitN.toInt, numFeatures)
  //      val amap = new mutable.HashMap[String, mutable.HashSet[Int]]()
  //      for (index <- iset) {
  //        val key = lsh.hashVector(train(index).features)
  //        val aset = amap.getOrElse(key, new mutable.HashSet[Int]())
  //        aset += index
  //        amap.update(key, aset)
  //      }
  //      val ans = new ListBuffer[mutable.HashSet[Int]]
  //      for (aset <- amap.values.toList) {
  //        val newlist = lshClusterRound(train, aset, numBits, numFeatures, perBucketN, setIterN, itN + 1)
  //        ans ++= newlist
  //      }
  //      ans.toList
  //    }
  //  }
  def lshClusterRound(
                       train: mutable.ArrayBuffer[LabeledPoint],
                       iset: mutable.HashSet[Int],
                       numFeatures: Int,
                       perBucketN: Int,
                       itN: Int
                     ): List[mutable.HashSet[Int]] = {
    var oldAns = new ListBuffer[mutable.HashSet[Int]]
    var ans = new ListBuffer[mutable.HashSet[Int]]
    oldAns += iset
    for (i <- 0 until itN) {
      print("lsh round:" + i + ",")
      ans.clear()
      for (oldset <- oldAns) {
        if (oldset.size <= perBucketN) {
          ans += oldset
        } else {
          val bN = oldset.size / perBucketN
          val bitN = math.log(bN) / math.log(2) + 1
          val lsh = new ZFLSH(bitN.toInt, numFeatures)
          val amap = new mutable.HashMap[String, mutable.HashSet[Int]]()
          for (index <- oldset) {
            val key = lsh.hashVector(train(index).features)
            val aset = amap.getOrElse(key, new mutable.HashSet[Int]())
            aset += index
            amap.update(key, aset)
          }
          ans ++= amap.values.toList
        }

      }
      oldAns.clear()
      oldAns ++= ans
    }
    println()
    ans.toList

  }


}
