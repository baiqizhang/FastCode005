package mapred.hashtagsim;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class SimilarityMapper extends Mapper<LongWritable, Text, IntWritable, Text> {

	Map<String, Integer> jobFeatures = null;

	/**
	 * We compute the inner product of feature vector of every hashtag with that
	 * of #job
	 */
	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String line = value.toString();
		String [] hashtags = line.split("\\|");



		String[] hashtag_featureVector = hashtags[0].split("\\s+");
		String hashtag = hashtag_featureVector[1];
		Map<String, Integer> features = parseFeatureVector(hashtag_featureVector[2]);

		String[] hashtag_featureVector1 = hashtags[1].split("\\s+");
		String hashtag1 = hashtag_featureVector1[0];
		Map<String, Integer> features1 = parseFeatureVector(hashtag_featureVector1[1]);

		Integer similarity = computeInnerProduct(features, features1);
		context.write(new IntWritable(similarity), new Text(hashtag  + "\t" + hashtag1));
	}


	/**
	 * De-serialize the feature vector into a map
	 * 
	 * @param featureVector
	 *            The format is "word1:count1;word2:count2;...;wordN:countN;"
	 * @return A HashMap, with key being each word and value being the count.
	 */
	private Map<String, Integer> parseFeatureVector(String featureVector) {
		Map<String, Integer> featureMap = new HashMap<String, Integer>();
		String[] features = featureVector.split(";");
		for (String feature : features) {
			String[] word_count = feature.split(":");
			featureMap.put(word_count[0], Integer.parseInt(word_count[1]));
		}
		return featureMap;
	}

	/**
	 * Computes the dot product of two feature vectors
	 * @param featureVector1
	 * @param featureVector2
	 * @return 
	 */
	private Integer computeInnerProduct(Map<String, Integer> featureVector1,
			Map<String, Integer> featureVector2) {
		Integer sum = 0;
		for (String word : featureVector1.keySet()) 
			if (featureVector2.containsKey(word))
				sum += featureVector1.get(word) * featureVector2.get(word);
		
		return sum;
	}
}














