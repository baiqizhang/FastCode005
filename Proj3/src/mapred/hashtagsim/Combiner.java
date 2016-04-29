package mapred.hashtagsim;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Created by LumiG on 3/16/16.
 */
public class Combiner extends Reducer<Text, IntWritable, Text, IntWritable>{
    @Override
    protected void reduce(Text key, Iterable<IntWritable> value,
                          Context context)
            throws IOException, InterruptedException {


        Integer count = 0;
        for (IntWritable c : value) {
            count += c.get();
        }

        context.write(new Text(key), new IntWritable(count));
    }
}



