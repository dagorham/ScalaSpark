package SparkMLExtension

import scala.util.parsing.json.JSON
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}
import org.apache.spark.sql.functions.{udf, col}


/**
  * Created by daniel on 6/21/17.
  */
class TweetParser(override val uid: String) extends Transformer {
  def getTweet(str: String): String = {
    try {
      JSON.parseFull(str) match {
        case Some(m: Map[String, String]) => return m("text")
      }
    } catch {
      case e: Exception => return "unknown"
    }
  }

  def getLang(str: String): Int = {
    try {
      val lang = JSON.parseFull(str) match {
        case Some(m: Map[String, String]) => m("lang")
      }

      if (lang == "en") return 1
      else if (lang == "es") return 0
      else return -1

    }
    catch {
      case e: Exception => -1
    }
  }

  override def copy(extra: ParamMap): TweetParser = {
    defaultCopy(extra)
  }

  override def transformSchema(schema: StructType): StructType = {
    val col_one = StructField("value", StringType, false)
    val col_two = StructField("text", StringType, false)
    val col_three = StructField("lang", IntegerType, false)

    StructType(Array(col_one, col_two, col_three))
  }

  override def transform(df: Dataset[_]): DataFrame  = {
    val tweet_udf = udf(getTweet _)
    val lang_udf = udf(getLang _)

    val df_1 = df.withColumn("text", tweet_udf(col("value")))
    val df_2 = df_1.withColumn("lang", lang_udf(col("value")))
    val df_3 = df_2.filter("lang != -1")

    return df_3
  }



}
