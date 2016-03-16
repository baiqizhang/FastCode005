package mapred.hashtagsim;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.*;

/**
 * Created by LumiG on 3/15/16.
 */
public class CartesianReducer extends Reducer<Text, Text, Text, Text> {
    @Override
    protected void reduce(Text key, Iterable<Text> value,
                          Context context)
            throws IOException, InterruptedException {


        Integer count = 0;
        for (Text c : value) {
            count += Integer.parseInt(c.toString());
        }

        context.write(new Text(count.toString()), new Text(key));
    }
}
