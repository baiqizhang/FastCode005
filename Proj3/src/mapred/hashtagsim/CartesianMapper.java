package mapred.hashtagsim;

import mapred.util.*;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by LumiG on 3/15/16.
 */
public class CartesianMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

    @Override
    protected void map(LongWritable key, Text value,
                       Context context)
            throws IOException, InterruptedException {


        String [] v = value.toString().split("\\s+");
        String [] v1 = v[1].split(";");
        int size = v1.length;
        String[] hashtag = new String[size];
        Integer[] existance = new Integer[size];
        for (int i = 0; i < size; i++){
                hashtag[i] = v1[i].split(":")[0];
                existance[i] = Integer.parseInt(v1[i].split(":")[1]);
        }

        for (int i = 0; i < size; i++){
            for (int j = i+1; j < size; j++){
                String outkey = hashtag[i] + "\t" + hashtag[j];
                int outvalue = existance[i] * existance[j];
                context.write(new Text(outkey), new IntWritable(outvalue));
            }
        }

    }
}


