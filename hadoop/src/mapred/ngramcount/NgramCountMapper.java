package mapred.ngramcount;

import java.io.IOException;

import mapred.util.Tokenizer;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;
import java.lang.StringBuilder;

public class NgramCountMapper extends Mapper<LongWritable, Text, Text, NullWritable> {

	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String line = value.toString();
		String[] words = Tokenizer.tokenize(line);

        Configuration conf = context.getConfiguration();
        int gramNum = Integer.parseInt(conf.get("gramNum"));
		
        for (int i = 0; i < words.length - gramNum + 1; i++){
            StringBuilder result = new StringBuilder();
            result.append(words[i]);
			for (int j = i+1; j < i+gramNum; j++){
                result.append(" ");
                result.append(words[j]);
            }
            context.write(new Text(result.toString()), NullWritable.get());
        }

	}
}


