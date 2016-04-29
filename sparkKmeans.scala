import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object sparkKmeans {
	def main(args: Array[String]) {
		val conf = new SparkConf().setAppName("Spark Kmeans")
		val sc = new SparkContext(conf)

		val rawData = sc.textFile("hdfs:///user/hadoop/features.txt", 16)
		val features = rawData.map(l => Vectors.dense(l.split(',').map(_.toDouble))).cache()

		// val rawData = sc.textFile("hdfs:///user/hadoop/output_hadoop.txt")
		// val rawFeatures = rawData.map(l => l.split('\t')(2))
		// val features = rawFeatures.map(l => Vectors.dense(l.slice(1,l.length-1).split(',').map(_.toDouble))).cache()

		val numClusters = 1000
		val numIterations = 2000
		val clusters = KMeans.train(features, numClusters, numIterations)

		val centroids = clusters.clusterCenters

		val imagesByCluster = rawData.map {
			l => 
			val rawFeatures = rawData.map(l => l.split('\t')(2))
			val features = rawFeatures.map(l => Vectors.dense(l.slice(1,l.length-1).split(',').map(_.toDouble))).cache()
			val key = l.split('\t').head
			(clusters.predict(features), key)
		}

		// imagesByCluster.saveAsTextFile("kmeansResult")

		// show_timing()
	}

	// def show_timing() = {
	//     val start=System.nanoTime()
	    
	//     val clusters = KMeans.train(features, numClusters, numIterations)

	//     val end = System.nanoTime()

	// 	println("Clustering time: " + (end-start)/1000000 + " millisecs")
	// }

	// def show_timing_10() = {
	// 	var times = Array[Long]()
	// 	for (i <- 0 until 10) {
	// 	    val start=System.nanoTime()
		    
	// 	    val clusters = KMeans.train(features, numClusters, numIterations)

	// 	    val end = System.nanoTime()
	// 	    times = times :+ (end-start)
	// 	}

	// 	var sum: Long = 0
	// 	var i = 0
	// 	while (i < times.length) {
	// 	  sum += times(i)
	// 	  i += 1
	// 	}		

	// 	println("Avg. clustering time: " + sum/1000000 / 10 + " millisecs")
	// }
}