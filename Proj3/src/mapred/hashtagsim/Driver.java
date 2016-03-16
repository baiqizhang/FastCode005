package mapred.hashtagsim;

import java.io.IOException;
import mapred.job.Optimizedjob;
import mapred.util.SimpleParser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

public class Driver {

	public static void main(String args[]) throws Exception {
		SimpleParser parser = new SimpleParser(args);

		String input = parser.get("input");
		String output = parser.get("output");
		String tmpdir = parser.get("tmpdir");

		// JobMapper, JobReducer
		// getJobFeatureVector(input, tmpdir + "/job_feature_vector");


		// String jobFeatureVector = loadJobFeatureVector(tmpdir
		//		+ "/job_feature_vector");

		//System.out.println("Job feature vector: " + jobFeatureVector);

		// Hashtag Mapper/ Reducer
		getHashtagFeatureVector(input, tmpdir + "/feature_vector");

        generateCartesian(tmpdir + "/feature_vector", output);

		// Similarity Mapper
		//getHashtagSimilarities(tmpdir + "/feature_vector",
		//		output);
	}

	/**
	 * Computes the word cooccurrence counts for hashtag #job
	 * 
	 * @param input
	 *            The directory of input files. It can be local directory, such
	 *            as "data/", "/home/ubuntu/data/", or Amazon S3 directory, such
	 *            as "s3n://myawesomedata/"
	 * @param output
	 *            Same format as input
	 * @throws IOException
	 * @throws ClassNotFoundException
	 * @throws InterruptedException
	 */
	private static void generateCartesian(String input, String output)
			throws IOException, ClassNotFoundException, InterruptedException {
		Optimizedjob job = new Optimizedjob(new Configuration(), input, output,
				"Get feature vector for hashtag #Job");

		job.setClasses(CartesianMapper.class, CartesianReducer.class, null);
		job.setMapOutputClasses(Text.class, Text.class);
		//job.setOutputKeyClass(Text.class);
		//job.setOutputValueClass(Text.class);
		//job.setReduceJobs(1);

		job.run();
	}



	/**
	 * Same as getJobFeatureVector, but this one actually computes feature
	 * vector for all hashtags.
	 * 
	 * @param input
	 * @param output
	 * @throws Exception
	 */
	private static void getHashtagFeatureVector(String input, String output)
			throws Exception {
		Optimizedjob job = new Optimizedjob(new Configuration(), input, output,
				"Get feature vector for all hashtags");
		job.setClasses(HashtagMapper.class, HashtagReducer.class, null);
		job.setMapOutputClasses(Text.class, Text.class);
		job.run();
	}

}