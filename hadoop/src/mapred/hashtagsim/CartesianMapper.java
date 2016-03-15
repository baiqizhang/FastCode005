package mapred.hashtagsim;

import mapred.util.*;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * Created by LumiG on 3/15/16.
 */
public class CartesianMapper extends Mapper<LongWritable, Text, Text, Text> {

    @Override
    protected void map(LongWritable key, Text value,
                       Context context)
            throws IOException, InterruptedException {

        context.write(new Text("k"), new Text(value.toString()));

    }
}

