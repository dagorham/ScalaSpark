/**
  * Created by daniel on 6/19/17.
  */

import SparkMLExtension.SparkContextCreator
import SparkMLExtension.TweetParser
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{RegexTokenizer, HashingTF, IDF}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

import scala.util.parsing.json._

object TwitterMVP {
  val tweetsPath = "file:///Users/daniel/IdeaProjects/ScalaSpark/data/Tweets/conor-twitterdata-1-2017-04-12-18-01-25-35b3cf72-d1d3-4462-a67a-dde73bea8c74.txt"

  def findVal(str: String, ToFind: String): String = {
    try {
      JSON.parseFull(str) match {
        case Some(m: Map[String, String]) => m(ToFind)
      }
    } catch {
      case e: Exception => null
    }
  }

  def getTweetsAndLang(input: String): (String, Int) = {
    try {
      val result = (findVal(input, "text"), -1)

      if (findVal(input, "lang") == "en") result.copy(_2 = 0)
      else if (findVal(input, "lang") == "es") result.copy(_2 = 1)
      else result
    } catch {
      case e: Exception => ("unknown", -1)
    }
  }

  def main(args: Array[String]) {
    val sc = SparkContextCreator.createSparkContext
    val sqlc = new SQLContext(sc)
    import sqlc.implicits._

    val df = sc.textFile(tweetsPath).toDF("value")

    val parser = new TweetParser("parser")

    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("words")
      .setPattern("""\s+|[,.\"]""")

    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(200)

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")

    val forestizer = new RandomForestClassifier()
      .setLabelCol("lang")
      .setFeaturesCol("features")
      .setNumTrees(10)

    val pipeline = new Pipeline()
      .setStages(Array(parser, tokenizer, hashingTF, idf, forestizer))

    val splitData = df.randomSplit(Array(.7, .3), seed=123)

    val (training, test) = (splitData(0), splitData(1))

    val model = pipeline.fit(training)

    val testModel = model.transform(test)

    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol("probability")
      .setLabelCol("lang")
      .setMetricName("areaUnderROC")

    val modelScore = evaluator.evaluate(testModel)

    println(modelScore)

    sc.stop()
  }
}