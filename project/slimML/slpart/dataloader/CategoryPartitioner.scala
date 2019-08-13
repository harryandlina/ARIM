package slpart.dataloader

import org.apache.spark.Partitioner

/**
  * partition by category
  * @param category
  */
class CategoryPartitioner(category: Int) extends Partitioner {
  override def numPartitions: Int = category

  override def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[Int]
    k % category
  }
}
