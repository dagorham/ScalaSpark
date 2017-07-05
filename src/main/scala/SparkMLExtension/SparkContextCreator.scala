package SparkMLExtension

/**
  * Created by daniel on 6/19/17.
  */

import org.apache.spark.{SparkConf, SparkContext}

object SparkContextCreator {
  def createSparkContext: SparkContext = {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    return sc
  }
}
