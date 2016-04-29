package mapred.hashtagsim;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.*;

/**
 * Created by LumiG on 3/15/16.
 */
public class CartesianReducer extends Reducer<Text, IntWritable, Text, Text> {
    @Override
    protected void reduce(Text key, Iterable<IntWritable> value,
                          Context context)
            throws IOException, InterruptedException {
        Integer count = 0;
        for (IntWritable c : value) {
            count += c.get();
        }
        context.write(new Text(count.toString()), new Text(key));
    }
}
