Hadoop
-------
To run hadoop job for image clustering, use commands listed in note.sh

Spark K-means
-------
To run spark kmeans, one should run 'sbt package' to build a .jar file from the .scala source code. (instructions at http://spark.apache.org/docs/latest/quick-start.html) Once successfully compiled, the .jar file can be submitted to a spark cluster using 'spark-submit --master yarn --deploy-mode cluster --driver-memory 2g --num-executors 8 --executor-cores 2 --executor-memory 4g --class sparkKmeans spark-kmeans_2.10-0.1.jar' The parameters can be tuned and the optimal tuning depends on the platform. Make sure to upload the data into hadoop first when running the source code without modification.

CNN
-------
To train a CNN, download images and use train_oxbuild.lua or train_GPU.lua. To run CNN with torch, use notes/init.sh for all nodes to install torch, copy dataset to hdfs, and use commands in note.sh to run the hadoop job. 
