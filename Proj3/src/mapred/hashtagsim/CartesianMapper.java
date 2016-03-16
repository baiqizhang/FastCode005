package mapred.hashtagsim;

import mapred.util.*;
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
public class CartesianMapper extends Mapper<LongWritable, Text, Text, Text> {

    @Override
    protected void map(LongWritable key, Text value,
                       Context context)
            throws IOException, InterruptedException {


        String [] v = value.toString().split("\\s+");

        List<String> hashtag = new ArrayList<>();
        List<Integer> existance = new ArrayList<>();
        for (String s: v[1].split(";")){
                hashtag.add(s.split(":")[0]);
                existance.add(Integer.parseInt(s.split(":")[1]));
        }

        for (int i = 0; i < hashtag.size(); i++){
            for (int j = i+1; j < hashtag.size(); j++){
                String outkey = hashtag.get(i) + "," + hashtag.get(j);
                Integer outvalue = existance.get(i) * existance.get(j);
                context.write(new Text(outkey), new Text(outvalue.toString()));
            }
        }

    }
}


